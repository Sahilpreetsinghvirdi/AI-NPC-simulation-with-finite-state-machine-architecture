#include "sim/ai/PersistentLearningPolicy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sim::ai {

PersistentLearningPolicy::PersistentLearningPolicy()
    : network_({.inputSize = 8, .hiddenSize = 16, .outputSize = 5, .seed = 1337})
    , trajectory_(32)
    , optimizer_({.learningRate = 0.0008f,
                  .discountFactor = 0.97f,
                  .gaeLambda = 0.95f,
                  .clipRange = 0.20f,
                  .valueLossCoefficient = 0.50f,
                  .entropyCoefficient = 0.01f,
                  .minibatchSize = 8,
                  .epochs = 4})
{
}

PolicyDescriptor PersistentLearningPolicy::GetDescriptor() const
{
    return {
        .id = "persistent_learning_fsm",
        .displayName = "Persistent Learning Policy with FSM Baseline",
        .supportsBatching = false,
        .supportsAsync = false,
        .deterministicByDefault = true
    };
}

void PersistentLearningPolicy::Reset()
{
    std::lock_guard lock(mutex_);
    baselinePolicy_.Reset();
}

PolicyResult PersistentLearningPolicy::Evaluate(const PolicyRequest& request)
{
    PolicyResult baselineResult = baselinePolicy_.Evaluate(request);
    if (!baselineResult.IsOk()) {
        return baselineResult;
    }

    std::lock_guard lock(mutex_);
    const std::vector<float> probabilities = network_.ForwardProbabilities(request.observation.features);
    const int learnedAction = network_.SelectGreedyAction(request.observation.features);
    const float confidence = learnedAction >= 0 && static_cast<std::size_t>(learnedAction) < probabilities.size()
                                 ? probabilities[static_cast<std::size_t>(learnedAction)]
                                 : 0.0f;

    baselineResult.decision.rawAction = sim::rl::Action::FromDiscrete(learnedAction);
    baselineResult.decision.confidence = confidence;

    if (trainingState_.learnedPolicyEnabled) {
        baselineResult.decision.action = static_cast<sim::entities::NpcAction>(learnedAction);
        baselineResult.decision.state = request.npcContext.playerWantedLevel > 0
                                            ? sim::entities::NpcState::Pursue
                                            : sim::entities::NpcState::Patrol;
    }

    return baselineResult;
}

TrainingResult PersistentLearningPolicy::ObserveTransition(const sim::rl::Transition& transition)
{
    std::lock_guard lock(mutex_);

    ++trainingState_.trainingStepCount;
    trainingState_.totalReward += transition.reward;
    lastTransition_ = transition;
    hasLastTransition_ = true;
    if (transition.terminal || transition.truncated) {
        ++trainingState_.episodeCount;
    }

    const int action = transition.action.discrete;
    const int clampedAction = std::clamp(action, 0, static_cast<int>(network_.GetConfig().outputSize) - 1);
    ActionValueEstimate& estimate = actionValues_[action];
    ++estimate.sampleCount;
    const float sampleCount = static_cast<float>(estimate.sampleCount);
    estimate.value += (transition.reward - estimate.value) / std::max(sampleCount, 1.0f);

    trajectory_.Add({
        .observation = transition.observation,
        .action = clampedAction,
        .reward = transition.reward,
        .oldLogProbability = network_.LogProbability(transition.observation.features, clampedAction),
        .value = network_.ForwardValue(transition.observation.features),
        .nextValue = network_.ForwardValue(transition.nextObservation.features),
        .terminal = transition.terminal,
        .truncated = transition.truncated
    });

    if (trajectory_.IsReady()) {
        const PpoUpdateStats stats = optimizer_.Update(network_, trajectory_);
        if (stats.updated) {
            ++trainingState_.policyUpdateCount;
            trainingState_.lastMeanReturn = stats.meanReturn;
            trainingState_.lastMeanAdvantage = stats.meanAdvantage;
            trainingState_.lastPolicyLoss = stats.meanPolicyLoss;
            trainingState_.lastValueLoss = stats.meanValueLoss;
            trainingState_.lastEntropy = stats.meanEntropy;
            trainingState_.lastTotalLoss = stats.meanTotalLoss;
            trainingState_.lastClipFraction = stats.meanClipFraction;
            trainingState_.lastKlDivergence = stats.meanApproxKlDivergence;
            trainingState_.lastExplainedVariance = stats.explainedVariance;
            trainingState_.lastGradientNorm = stats.meanGradientNorm;
            trainingState_.lastParameterUpdateNorm = stats.meanParameterUpdateNorm;
            trainingState_.lastMaxParameterUpdate = stats.maxParameterUpdate;
            trainingState_.lastReturn = stats.latestReturn;
            trainingState_.lastAdvantage = stats.latestAdvantage;
            trainingState_.lastPpoEpoch = stats.lastEpoch;
            trainingState_.lastPpoMinibatch = stats.lastMinibatch;
        }
        trajectory_.Clear();
    }

    return {
        .updated = true,
        .state = trainingState_
    };
}

TrainingState PersistentLearningPolicy::GetTrainingState() const
{
    std::lock_guard lock(mutex_);
    return trainingState_;
}

PolicyCheckpoint PersistentLearningPolicy::ExportCheckpoint() const
{
    std::lock_guard lock(mutex_);
    return {
        .formatVersion = 2,
        .trainingState = trainingState_,
        .actionValues = actionValues_,
        .network = {
            .inputSize = network_.GetConfig().inputSize,
            .hiddenSize = network_.GetConfig().hiddenSize,
            .outputSize = network_.GetConfig().outputSize,
            .parameters = network_.ExportParameters()
        }
    };
}

AiDebugSnapshot PersistentLearningPolicy::BuildDebugSnapshot(const std::uint64_t simulationTick,
                                                             const float elapsedSeconds,
                                                             const float fps,
                                                             const std::size_t activePolice) const
{
    std::lock_guard lock(mutex_);

    AiDebugSnapshot snapshot;
    snapshot.simulationTick = simulationTick;
    snapshot.elapsedSeconds = elapsedSeconds;
    snapshot.fps = fps;
    snapshot.activePolice = activePolice;
    snapshot.trainingState = trainingState_;
    snapshot.rolloutSize = trajectory_.Size();
    snapshot.rolloutCapacity = trajectory_.Capacity();
    snapshot.learningRate = optimizer_.GetConfig().learningRate;
    snapshot.clipRange = optimizer_.GetConfig().clipRange;
    snapshot.entropyCoefficient = optimizer_.GetConfig().entropyCoefficient;
    snapshot.lastReturn = trainingState_.lastReturn;
    snapshot.lastAdvantage = trainingState_.lastAdvantage;

    if (hasLastTransition_) {
        snapshot.lastAction = lastTransition_.action.discrete;
        snapshot.lastReward = lastTransition_.reward;
        snapshot.observation = lastTransition_.observation.features;
        const NetworkEvaluation evaluation = network_.Evaluate(lastTransition_.observation.features);
        const NetworkEvaluation nextEvaluation = network_.Evaluate(lastTransition_.nextObservation.features);
        snapshot.normalizedObservation = evaluation.normalizedInput;
        snapshot.hiddenActivations = evaluation.hiddenActivations;
        snapshot.actorLogits = evaluation.logits;
        snapshot.actionProbabilities = evaluation.probabilities;
        snapshot.lastValue = evaluation.value;
        snapshot.lastNextValue = nextEvaluation.value;
        snapshot.lastTdError = lastTransition_.reward + (0.97f * nextEvaluation.value) - evaluation.value;
        snapshot.lastLogProbability = network_.LogProbability(lastTransition_.observation.features, snapshot.lastAction);
    }

    snapshot.networkParameters = network_.ExportParameters();
    if (!snapshot.networkParameters.empty()) {
        snapshot.weightStats.minWeight = std::numeric_limits<float>::max();
        snapshot.weightStats.maxWeight = std::numeric_limits<float>::lowest();
        float absSum = 0.0f;
        float squareSum = 0.0f;
        for (const float parameter : snapshot.networkParameters) {
            snapshot.weightStats.minWeight = std::min(snapshot.weightStats.minWeight, parameter);
            snapshot.weightStats.maxWeight = std::max(snapshot.weightStats.maxWeight, parameter);
            absSum += std::fabs(parameter);
            squareSum += parameter * parameter;
        }
        snapshot.weightStats.meanAbsWeight = absSum / static_cast<float>(snapshot.networkParameters.size());
        snapshot.weightStats.l2Norm = std::sqrt(squareSum);
        snapshot.weightStats.parameterDrift = trainingState_.lastParameterUpdateNorm;
        snapshot.weightStats.maxParameterUpdate = trainingState_.lastMaxParameterUpdate;
    }

    return snapshot;
}

bool PersistentLearningPolicy::ImportCheckpoint(const PolicyCheckpoint& checkpoint)
{
    if (checkpoint.formatVersion != 1 && checkpoint.formatVersion != 2) {
        return false;
    }

    std::lock_guard lock(mutex_);
    trainingState_ = checkpoint.trainingState;
    actionValues_ = checkpoint.actionValues;
    if (checkpoint.formatVersion >= 2 && !checkpoint.network.parameters.empty()) {
        if (checkpoint.network.inputSize != network_.GetConfig().inputSize ||
            checkpoint.network.hiddenSize != network_.GetConfig().hiddenSize ||
            checkpoint.network.outputSize != network_.GetConfig().outputSize ||
            !network_.ImportParameters(checkpoint.network.parameters)) {
            return false;
        }
    }
    return true;
}

} // namespace sim::ai
