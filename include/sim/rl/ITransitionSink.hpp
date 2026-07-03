#pragma once

#include "sim/rl/RlTypes.hpp"

namespace sim::rl {

class ITransitionSink {
public:
    virtual ~ITransitionSink() = default;

    virtual void BeginEpisode(EpisodeId episodeId) = 0;
    virtual void RecordTransition(const Transition& transition) = 0;
    virtual void EndEpisode(const EpisodeSummary& summary) = 0;
};

class NullTransitionSink final : public ITransitionSink {
public:
    void BeginEpisode(EpisodeId episodeId) override;
    void RecordTransition(const Transition& transition) override;
    void EndEpisode(const EpisodeSummary& summary) override;
};

} // namespace sim::rl
