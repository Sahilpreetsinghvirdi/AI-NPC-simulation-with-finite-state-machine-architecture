#pragma once

#include "sim/ai/IPolicy.hpp"
#include "sim/rl/RlTypes.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim::ai {

struct ActionValueEstimate {
    float value{0.0f};
    std::uint64_t sampleCount{0};
};

struct TrainingState {
    std::uint64_t trainingStepCount{0};
    std::uint64_t episodeCount{0};
    float totalReward{0.0f};
    std::uint64_t policyUpdateCount{0};
    float lastMeanReturn{0.0f};
    float lastMeanAdvantage{0.0f};
    bool learnedPolicyEnabled{false};
    std::string algorithmId{"ppo_ready_policy_gradient"};
    std::string policyId{"persistent_learning_fsm"};
};

struct NetworkCheckpoint {
    std::size_t inputSize{0};
    std::size_t hiddenSize{0};
    std::size_t outputSize{0};
    std::vector<float> parameters;
};

struct PolicyCheckpoint {
    std::uint32_t formatVersion{2};
    TrainingState trainingState;
    std::unordered_map<int, ActionValueEstimate> actionValues;
    NetworkCheckpoint network;
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
