#include "sim/rl/EpisodeRecorder.hpp"

#include <algorithm>

namespace sim::rl {

void NullTransitionSink::BeginEpisode(const EpisodeId /*episodeId*/)
{
}

void NullTransitionSink::RecordTransition(const Transition& /*transition*/)
{
}

void NullTransitionSink::EndEpisode(const EpisodeSummary& /*summary*/)
{
}

EpisodeRecorder::EpisodeRecorder()
    : EpisodeRecorder(Config{})
{
}

EpisodeRecorder::EpisodeRecorder(Config config)
    : config_(config)
{
}

bool EpisodeRecorder::IsEnabled() const
{
    return config_.mode != RunMode::Disabled;
}

RunMode EpisodeRecorder::GetMode() const
{
    return config_.mode;
}

EpisodeId EpisodeRecorder::GetCurrentEpisodeId() const
{
    return currentEpisodeId_;
}

std::size_t EpisodeRecorder::GetTransitionCount() const
{
    return transitionCount_;
}

float EpisodeRecorder::GetTotalReward() const
{
    return totalReward_;
}

const std::vector<Transition>& EpisodeRecorder::GetTransitions() const
{
    return transitions_;
}

void EpisodeRecorder::BeginEpisode(const EpisodeId episodeId)
{
    currentEpisodeId_ = episodeId;
    Clear();
}

void EpisodeRecorder::RecordTransition(const Transition& transition)
{
    if (!IsEnabled()) {
        return;
    }

    ++transitionCount_;
    totalReward_ += transition.reward;

    if (!config_.keepTransitionsInMemory || config_.maxStoredTransitions == 0) {
        return;
    }

    if (transitions_.size() >= config_.maxStoredTransitions) {
        transitions_.erase(transitions_.begin());
    }

    transitions_.push_back(transition);
}

void EpisodeRecorder::EndEpisode(const EpisodeSummary& summary)
{
    currentEpisodeId_ = summary.episodeId;
    transitionCount_ = summary.transitionCount;
    totalReward_ = summary.totalReward;
}

void EpisodeRecorder::Clear()
{
    transitions_.clear();
    transitionCount_ = 0;
    totalReward_ = 0.0f;
}

} // namespace sim::rl
