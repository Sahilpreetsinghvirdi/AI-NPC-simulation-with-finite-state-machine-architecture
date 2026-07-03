#include "sim/ai/PpoOptimizer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sim::ai {

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
    float runningReturn = 0.0f;
    for (std::size_t reverseIndex = steps.size(); reverseIndex > 0; --reverseIndex) {
        const std::size_t index = reverseIndex - 1;
        if (steps[index].terminal || steps[index].truncated) {
            runningReturn = 0.0f;
        }
        runningReturn = steps[index].reward + (config_.discountFactor * runningReturn);
        returns[index] = runningReturn;
    }

    float meanReturn = 0.0f;
    for (const float value : returns) {
        meanReturn += value;
    }
    meanReturn /= static_cast<float>(returns.size());

    float meanAbsAdvantage = 0.0f;
    for (std::size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        for (std::size_t index = 0; index < steps.size(); ++index) {
            const float advantage = returns[index] - meanReturn;
            meanAbsAdvantage += std::fabs(advantage);

            const float newLogProbability = network.LogProbability(steps[index].observation.features, steps[index].action);
            const float ratio = std::exp(newLogProbability - steps[index].oldLogProbability);
            const float clippedRatio = std::clamp(ratio, 1.0f - config_.clipRange, 1.0f + config_.clipRange);
            const float clippedAdvantage = advantage * clippedRatio;

            network.ApplyPolicyGradient(steps[index].observation.features,
                                        steps[index].action,
                                        clippedAdvantage,
                                        config_.learningRate);
        }
    }

    meanAbsAdvantage /= static_cast<float>(steps.size() * std::max<std::size_t>(config_.epochs, 1));
    return {
        .updated = true,
        .sampleCount = steps.size(),
        .meanReturn = meanReturn,
        .meanAdvantage = meanAbsAdvantage
    };
}

} // namespace sim::ai
