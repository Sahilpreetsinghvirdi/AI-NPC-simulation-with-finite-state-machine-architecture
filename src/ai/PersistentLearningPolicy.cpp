#include "sim/ai/PersistentLearningPolicy.hpp"

#include <algorithm>

namespace sim::ai {

PersistentLearningPolicy::PersistentLearningPolicy() = default;

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
    return baselinePolicy_.Evaluate(request);
}

TrainingResult PersistentLearningPolicy::ObserveTransition(const sim::rl::Transition& transition)
{
    std::lock_guard lock(mutex_);

    ++trainingState_.trainingStepCount;
    trainingState_.totalReward += transition.reward;
    if (transition.terminal || transition.truncated) {
        ++trainingState_.episodeCount;
    }

    const int action = transition.action.discrete;
    ActionValueEstimate& estimate = actionValues_[action];
    ++estimate.sampleCount;
    const float sampleCount = static_cast<float>(estimate.sampleCount);
    estimate.value += (transition.reward - estimate.value) / std::max(sampleCount, 1.0f);

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
        .formatVersion = 1,
        .trainingState = trainingState_,
        .actionValues = actionValues_
    };
}

bool PersistentLearningPolicy::ImportCheckpoint(const PolicyCheckpoint& checkpoint)
{
    if (checkpoint.formatVersion != 1) {
        return false;
    }

    std::lock_guard lock(mutex_);
    trainingState_ = checkpoint.trainingState;
    actionValues_ = checkpoint.actionValues;
    return true;
}

} // namespace sim::ai
