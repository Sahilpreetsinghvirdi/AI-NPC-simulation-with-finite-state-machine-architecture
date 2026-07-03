#pragma once

#include "sim/rl/ITransitionSink.hpp"
#include "sim/rl/RlTypes.hpp"

#include <cstddef>
#include <vector>

namespace sim::rl {

class EpisodeRecorder final : public ITransitionSink {
public:
    struct Config {
        RunMode mode{RunMode::Training};
        std::size_t maxStoredTransitions{4096};
        bool keepTransitionsInMemory{true};
    };

    EpisodeRecorder();
    explicit EpisodeRecorder(Config config);

    [[nodiscard]] bool IsEnabled() const;
    [[nodiscard]] RunMode GetMode() const;
    [[nodiscard]] EpisodeId GetCurrentEpisodeId() const;
    [[nodiscard]] std::size_t GetTransitionCount() const;
    [[nodiscard]] float GetTotalReward() const;
    [[nodiscard]] const std::vector<Transition>& GetTransitions() const;

    void BeginEpisode(EpisodeId episodeId) override;
    void RecordTransition(const Transition& transition) override;
    void EndEpisode(const EpisodeSummary& summary) override;
    void Clear();

private:
    Config config_;
    EpisodeId currentEpisodeId_{0};
    std::vector<Transition> transitions_;
    std::size_t transitionCount_{0};
    float totalReward_{0.0f};
};

} // namespace sim::rl
