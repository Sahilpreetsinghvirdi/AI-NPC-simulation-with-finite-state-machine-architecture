#pragma once

#include "sim/ai/Observation.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sim::rl {

using AgentId = std::uint32_t;
using EpisodeId = std::uint64_t;
using StepIndex = std::uint64_t;

enum class RunMode {
    Disabled = 0,
    Training,
    Evaluation,
    Inference
};

enum class ActionSpaceType {
    Discrete = 0,
    Continuous,
    Hybrid
};

struct AgentHandle {
    AgentId id{0};
    std::string role;
};

struct Action {
    ActionSpaceType type{ActionSpaceType::Discrete};
    int discrete{0};
    std::vector<float> continuous;

    [[nodiscard]] static Action FromDiscrete(int value)
    {
        return {
            .type = ActionSpaceType::Discrete,
            .discrete = value,
            .continuous = {}
        };
    }
};

struct StepMetadata {
    EpisodeId episodeId{0};
    StepIndex stepIndex{0};
    float elapsedSeconds{0.0f};
};

struct Transition {
    AgentHandle agent;
    sim::ai::Observation observation;
    Action action;
    float reward{0.0f};
    sim::ai::Observation nextObservation;
    bool terminal{false};
    bool truncated{false};
    StepMetadata metadata;
};

struct EpisodeSummary {
    EpisodeId episodeId{0};
    std::size_t transitionCount{0};
    float totalReward{0.0f};
    bool terminal{false};
    bool truncated{false};
};

} // namespace sim::rl
