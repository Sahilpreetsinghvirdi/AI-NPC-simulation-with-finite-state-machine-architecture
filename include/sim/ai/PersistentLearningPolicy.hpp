#pragma once

#include "sim/ai/AiDebugSnapshot.hpp"
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
    [[nodiscard]] AiDebugSnapshot BuildDebugSnapshot(std::uint64_t simulationTick,
                                                     float elapsedSeconds,
                                                     float fps,
                                                     std::size_t activePolice) const;
    bool ImportCheckpoint(const PolicyCheckpoint& checkpoint) override;

private:
    mutable std::mutex mutex_;
    FsmPolicy baselinePolicy_;
    FeedForwardNetwork network_;
    PpoTrajectoryBuffer trajectory_;
    PpoOptimizer optimizer_;
    TrainingState trainingState_;
    std::unordered_map<int, ActionValueEstimate> actionValues_;
    sim::rl::Transition lastTransition_;
    bool hasLastTransition_{false};
};

} // namespace sim::ai
