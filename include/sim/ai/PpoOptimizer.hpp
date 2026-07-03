#pragma once

#include "sim/ai/FeedForwardNetwork.hpp"
#include "sim/ai/PpoTrajectoryBuffer.hpp"

#include <cstddef>

namespace sim::ai {

struct PpoOptimizerConfig {
    float learningRate{0.0008f};
    float discountFactor{0.97f};
    float clipRange{0.20f};
    std::size_t epochs{2};
};

struct PpoUpdateStats {
    bool updated{false};
    std::size_t sampleCount{0};
    float meanReturn{0.0f};
    float meanAdvantage{0.0f};
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
