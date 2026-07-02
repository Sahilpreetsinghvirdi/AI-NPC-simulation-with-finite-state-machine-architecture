#include "sim/core/SimulationTimer.hpp"

#include <stdexcept>
#include <thread>

namespace sim::core {

SimulationTimer::SimulationTimer()
    : SimulationTimer(Config{})
{
}

SimulationTimer::SimulationTimer(const Config config)
    : config_(config)
{
    ValidateTickRate();
    deltaTimeSeconds_ = 1.0f / config_.tickRateHz;
    nextTickDeadline_ = std::chrono::steady_clock::now();
}

void SimulationTimer::Reset()
{
    elapsedTimeSeconds_ = 0.0f;
    tickIndex_ = 0;
    pacingInitialized_ = false;
    nextTickDeadline_ = std::chrono::steady_clock::now();
}

void SimulationTimer::Advance()
{
    if (config_.realTimePacing) {
        PaceToNextTick();
    }

    ++tickIndex_;
    elapsedTimeSeconds_ += deltaTimeSeconds_;
}

void SimulationTimer::SetTickRateHz(const float tickRateHz)
{
    config_.tickRateHz = tickRateHz;
    ValidateTickRate();
    deltaTimeSeconds_ = 1.0f / config_.tickRateHz;
}

void SimulationTimer::SetRealTimePacing(const bool enabled)
{
    config_.realTimePacing = enabled;
    pacingInitialized_ = false;
    nextTickDeadline_ = std::chrono::steady_clock::now();
}

float SimulationTimer::GetDeltaTime() const
{
    return deltaTimeSeconds_;
}

float SimulationTimer::GetElapsedTime() const
{
    return elapsedTimeSeconds_;
}

std::uint64_t SimulationTimer::GetTickIndex() const
{
    return tickIndex_;
}

float SimulationTimer::GetTickRateHz() const
{
    return config_.tickRateHz;
}

bool SimulationTimer::IsRealTimePacingEnabled() const
{
    return config_.realTimePacing;
}

void SimulationTimer::ValidateTickRate() const
{
    if (config_.tickRateHz <= 0.0f) {
        throw std::invalid_argument("SimulationTimer tick rate must be greater than zero.");
    }
}

void SimulationTimer::PaceToNextTick()
{
    const auto now = std::chrono::steady_clock::now();

    if (!pacingInitialized_) {
        nextTickDeadline_ = now;
        pacingInitialized_ = true;
    }

    const auto tickDuration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<float>(deltaTimeSeconds_));

    nextTickDeadline_ += tickDuration;

    if (nextTickDeadline_ > now) {
        std::this_thread::sleep_until(nextTickDeadline_);
    } else {
        // Simulation is behind real time; resync instead of trying to catch up with burst sleeps.
        nextTickDeadline_ = now;
    }
}

} // namespace sim::core
