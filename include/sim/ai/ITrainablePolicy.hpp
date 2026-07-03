#pragma once

#include "sim/ai/IPolicy.hpp"
#include "sim/rl/RlTypes.hpp"

#include <string>
#include <unordered_map>

namespace sim::ai {

struct ActionValueEstimate {
    float value{0.0f};
    std::uint64_t sampleCount{0};
};

struct TrainingState {
    std::uint64_t trainingStepCount{0};
    std::uint64_t episodeCount{0};
    float totalReward{0.0f};
    std::string algorithmId{"incremental_action_value"};
    std::string policyId{"persistent_learning_fsm"};
};

struct PolicyCheckpoint {
    std::uint32_t formatVersion{1};
    TrainingState trainingState;
    std::unordered_map<int, ActionValueEstimate> actionValues;
};

struct TrainingResult {
    bool updated{false};
    TrainingState state;
};

class ITrainablePolicy : public IPolicy {
public:
    ~ITrainablePolicy() override = default;

    virtual TrainingResult ObserveTransition(const sim::rl::Transition& transition) = 0;
    [[nodiscard]] virtual TrainingState GetTrainingState() const = 0;
    [[nodiscard]] virtual PolicyCheckpoint ExportCheckpoint() const = 0;
    virtual bool ImportCheckpoint(const PolicyCheckpoint& checkpoint) = 0;
};

} // namespace sim::ai
