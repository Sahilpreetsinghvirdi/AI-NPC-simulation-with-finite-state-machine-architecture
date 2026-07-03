#pragma once

#include "sim/ai/FeedForwardNetwork.hpp"
#include "sim/ai/FsmPolicy.hpp"
#include "sim/ai/ITrainablePolicy.hpp"
#include "sim/ai/PpoOptimizer.hpp"
#include "sim/ai/PpoTrajectoryBuffer.hpp"

#include <mutex>
#include <unordered_map>

namespace sim::ai {

class PersistentLearningPolicy final : public ITrainablePolicy {
public:
    PersistentLearningPolicy();

    [[nodiscard]] PolicyDescriptor GetDescriptor() const override;
    void Reset() override;
    [[nodiscard]] PolicyResult Evaluate(const PolicyRequest& request) override;

    TrainingResult ObserveTransition(const sim::rl::Transition& transition) override;
    [[nodiscard]] TrainingState GetTrainingState() const override;
    [[nodiscard]] PolicyCheckpoint ExportCheckpoint() const override;
    bool ImportCheckpoint(const PolicyCheckpoint& checkpoint) override;

private:
    mutable std::mutex mutex_;
    FsmPolicy baselinePolicy_;
    FeedForwardNetwork network_;
    PpoTrajectoryBuffer trajectory_;
    PpoOptimizer optimizer_;
    TrainingState trainingState_;
    std::unordered_map<int, ActionValueEstimate> actionValues_;
};

} // namespace sim::ai
