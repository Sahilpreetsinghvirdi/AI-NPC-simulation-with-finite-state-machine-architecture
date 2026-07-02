#pragma once

#include <string>
#include <vector>

namespace sim::ai {

// Normalized feature vector consumed by future RL policies and training pipelines.
struct Observation {
    std::vector<float> features;

    [[nodiscard]] std::size_t Size() const { return features.size(); }
    [[nodiscard]] std::string ToString() const;
};

} // namespace sim::ai
