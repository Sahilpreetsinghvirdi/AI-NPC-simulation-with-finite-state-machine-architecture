#pragma once

#include "sim/ai/FeedForwardNetwork.hpp"
#include "sim/ai/PpoTrajectoryBuffer.hpp"

#include <cstddef>

namespace sim::ai {

struct PpoOptimizerConfig {
    float learningRate{0.0008f};
    float discountFactor{0.97f};
    float gaeLambda{0.95f};
    float clipRange{0.20f};
    float valueLossCoefficient{0.50f};
    float entropyCoefficient{0.01f};
    std::size_t minibatchSize{8};
    std::size_t epochs{2};
};

struct PpoUpdateStats {
    bool updated{false};
    std::size_t sampleCount{0};
    float meanReturn{0.0f};
    float meanAdvantage{0.0f};
    float meanPolicyLoss{0.0f};
    float meanValueLoss{0.0f};
    float meanEntropy{0.0f};
    float meanTotalLoss{0.0f};
    float meanClipFraction{0.0f};
    float meanApproxKlDivergence{0.0f};
    float explainedVariance{0.0f};
    float meanGradientNorm{0.0f};
    float meanParameterUpdateNorm{0.0f};
    float maxParameterUpdate{0.0f};
    float latestReturn{0.0f};
    float latestAdvantage{0.0f};
    std::size_t lastEpoch{0};
    std::size_t lastMinibatch{0};
};

class PpoOptimizer {
public:
    explicit PpoOptimizer(PpoOptimizerConfig config = {});

    [[nodiscard]] const PpoOptimizerConfig& GetConfig() const;
    [[nodiscard]] PpoUpdateStats Update(FeedForwardNetwork& network, const PpoTrajectoryBuffer& trajectory) const;

private:
    PpoOptimizerConfig config_;
};

} // namespace sim::ai
