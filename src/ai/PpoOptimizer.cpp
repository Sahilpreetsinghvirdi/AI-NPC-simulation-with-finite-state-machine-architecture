#include "sim/ai/PpoOptimizer.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sim::ai {

namespace {

float VarianceOf(const std::vector<float>& values)
{
    if (values.empty()) {
        return 0.0f;
    }

    const float mean = std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
    float variance = 0.0f;
    for (const float value : values) {
        const float delta = value - mean;
        variance += delta * delta;
    }
    return variance / static_cast<float>(values.size());
}

} // namespace

PpoOptimizer::PpoOptimizer(PpoOptimizerConfig config)
    : config_(config)
{
}

const PpoOptimizerConfig& PpoOptimizer::GetConfig() const
{
    return config_;
}

PpoUpdateStats PpoOptimizer::Update(FeedForwardNetwork& network, const PpoTrajectoryBuffer& trajectory) const
{
    if (trajectory.Steps().empty()) {
        return {};
    }

    const std::vector<PpoTrajectoryStep>& steps = trajectory.Steps();
    std::vector<float> returns(steps.size(), 0.0f);
    std::vector<float> advantages(steps.size(), 0.0f);
    float runningAdvantage = 0.0f;
    for (std::size_t reverseIndex = steps.size(); reverseIndex > 0; --reverseIndex) {
        const std::size_t index = reverseIndex - 1;
        const float nonTerminal = (steps[index].terminal || steps[index].truncated) ? 0.0f : 1.0f;
        const float delta = steps[index].reward +
                            (config_.discountFactor * steps[index].nextValue * nonTerminal) -
                            steps[index].value;
        runningAdvantage = delta + (config_.discountFactor * config_.gaeLambda * nonTerminal * runningAdvantage);
        advantages[index] = runningAdvantage;
        returns[index] = advantages[index] + steps[index].value;
    }

    const float meanReturn = std::accumulate(returns.begin(), returns.end(), 0.0f) /
                             static_cast<float>(returns.size());

    float meanAdvantage = 0.0f;
    for (const float advantage : advantages) {
        meanAdvantage += advantage;
    }
    meanAdvantage /= static_cast<float>(advantages.size());

    float advantageVariance = 0.0f;
    for (float& advantage : advantages) {
        advantageVariance += (advantage - meanAdvantage) * (advantage - meanAdvantage);
    }
    advantageVariance /= static_cast<float>(advantages.size());
    const float advantageStdDev = std::sqrt(std::max(advantageVariance, 1.0e-6f));
    for (float& advantage : advantages) {
        advantage = (advantage - meanAdvantage) / advantageStdDev;
    }

    const std::size_t minibatchSize = std::max<std::size_t>(1, config_.minibatchSize);
    float accumulatedPolicyLoss = 0.0f;
    float accumulatedValueLoss = 0.0f;
    float accumulatedEntropy = 0.0f;
    float accumulatedTotalLoss = 0.0f;
    float accumulatedClipFraction = 0.0f;
    float accumulatedApproxKl = 0.0f;
    float accumulatedGradientNorm = 0.0f;
    float accumulatedParameterUpdateNorm = 0.0f;
    float maxParameterUpdate = 0.0f;
    std::size_t updateCount = 0;
    std::size_t lastEpoch = 0;
    std::size_t lastMinibatch = 0;
    std::vector<float> valuePredictions;
    valuePredictions.reserve(steps.size());
    for (const PpoTrajectoryStep& step : steps) {
        valuePredictions.push_back(step.value);
    }

    for (std::size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        for (std::size_t batchStart = 0; batchStart < steps.size(); batchStart += minibatchSize) {
            const std::size_t batchEnd = std::min(batchStart + minibatchSize, steps.size());
            lastEpoch = epoch + 1;
            lastMinibatch = (batchStart / minibatchSize) + 1;
            for (std::size_t index = batchStart; index < batchEnd; ++index) {
                const NetworkEvaluation evaluation = network.Evaluate(steps[index].observation.features);
                const float newLogProbability = network.LogProbability(steps[index].observation.features, steps[index].action);
                const float logRatio = newLogProbability - steps[index].oldLogProbability;
                const float ratio = std::exp(logRatio);
                const float clippedRatio = std::clamp(ratio, 1.0f - config_.clipRange, 1.0f + config_.clipRange);
                const float unclippedObjective = ratio * advantages[index];
                const float clippedObjective = clippedRatio * advantages[index];
                const bool useClippedGradient = clippedObjective < unclippedObjective;
                const float policyGradientScale = (useClippedGradient ? clippedRatio : ratio) * advantages[index];
                const float valueError = returns[index] - evaluation.value;

                const GradientUpdateStats gradientStats =
                    network.ApplyActorCriticGradient(steps[index].observation.features,
                                                     steps[index].action,
                                                     policyGradientScale,
                                                     returns[index],
                                                     config_.valueLossCoefficient,
                                                     config_.entropyCoefficient,
                                                     config_.learningRate);

                const float policyLoss = -std::min(unclippedObjective, clippedObjective);
                const float valueLoss = 0.5f * valueError * valueError;
                accumulatedPolicyLoss += policyLoss;
                accumulatedValueLoss += valueLoss;
                accumulatedEntropy += evaluation.entropy;
                accumulatedTotalLoss += policyLoss +
                                        (config_.valueLossCoefficient * valueLoss) -
                                        (config_.entropyCoefficient * evaluation.entropy);
                accumulatedClipFraction += (std::fabs(ratio - 1.0f) > config_.clipRange) ? 1.0f : 0.0f;
                accumulatedApproxKl += (ratio - 1.0f) - logRatio;
                accumulatedGradientNorm += gradientStats.gradientL2Norm;
                accumulatedParameterUpdateNorm += gradientStats.parameterUpdateL2Norm;
                maxParameterUpdate = std::max(maxParameterUpdate, gradientStats.maxParameterUpdate);
                ++updateCount;
            }
        }
    }

    float meanAbsAdvantage = 0.0f;
    for (const float advantage : advantages) {
        meanAbsAdvantage += std::fabs(advantage);
    }
    meanAbsAdvantage /= static_cast<float>(advantages.size());

    const float denominator = static_cast<float>(std::max<std::size_t>(updateCount, 1));
    const float returnVariance = VarianceOf(returns);
    float residualVariance = 0.0f;
    for (std::size_t index = 0; index < returns.size(); ++index) {
        const float residual = returns[index] - valuePredictions[index];
        residualVariance += residual * residual;
    }
    residualVariance /= static_cast<float>(returns.size());
    return {
        .updated = true,
        .sampleCount = steps.size(),
        .meanReturn = meanReturn,
        .meanAdvantage = meanAbsAdvantage,
        .meanPolicyLoss = accumulatedPolicyLoss / denominator,
        .meanValueLoss = accumulatedValueLoss / denominator,
        .meanEntropy = accumulatedEntropy / denominator,
        .meanTotalLoss = accumulatedTotalLoss / denominator,
        .meanClipFraction = accumulatedClipFraction / denominator,
        .meanApproxKlDivergence = accumulatedApproxKl / denominator,
        .explainedVariance = returnVariance > 1.0e-6f ? 1.0f - (residualVariance / returnVariance) : 0.0f,
        .meanGradientNorm = accumulatedGradientNorm / denominator,
        .meanParameterUpdateNorm = accumulatedParameterUpdateNorm / denominator,
        .maxParameterUpdate = maxParameterUpdate,
        .latestReturn = returns.back(),
        .latestAdvantage = advantages.back(),
        .lastEpoch = lastEpoch,
        .lastMinibatch = lastMinibatch
    };
}

} // namespace sim::ai
