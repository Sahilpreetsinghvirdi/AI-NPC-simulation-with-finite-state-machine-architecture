#pragma once

#include "sim/ai/Observation.hpp"

#include <cstddef>
#include <vector>

namespace sim::ai {

struct PpoTrajectoryStep {
    Observation observation;
    int action{0};
    float reward{0.0f};
    float oldLogProbability{0.0f};
    bool terminal{false};
    bool truncated{false};
};

class PpoTrajectoryBuffer {
public:
    explicit PpoTrajectoryBuffer(std::size_t capacity = 64);

    void Add(PpoTrajectoryStep step);
    void Clear();

    [[nodiscard]] bool IsReady() const;
    [[nodiscard]] std::size_t Size() const;
    [[nodiscard]] std::size_t Capacity() const;
    [[nodiscard]] const std::vector<PpoTrajectoryStep>& Steps() const;

private:
    std::size_t capacity_{64};
    std::vector<PpoTrajectoryStep> steps_;
};

} // namespace sim::ai
