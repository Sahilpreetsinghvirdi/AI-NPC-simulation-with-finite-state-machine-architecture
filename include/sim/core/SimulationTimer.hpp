#pragma once

#include <chrono>
#include <cstdint>

namespace sim::core {

// Fixed-step simulation timer. Provides deterministic dt for gameplay and future RL stepping.
class SimulationTimer {
public:
    struct Config {
        float tickRateHz{10.0f};
        bool realTimePacing{false};
    };

    SimulationTimer();
    explicit SimulationTimer(Config config);

    void Reset();
    void Advance();

    void SetTickRateHz(float tickRateHz);
    void SetRealTimePacing(bool enabled);

    [[nodiscard]] float GetDeltaTime() const;
    [[nodiscard]] float GetElapsedTime() const;
    [[nodiscard]] std::uint64_t GetTickIndex() const;
    [[nodiscard]] float GetTickRateHz() const;
    [[nodiscard]] bool IsRealTimePacingEnabled() const;

private:
    void ValidateTickRate() const;
    void PaceToNextTick();

    Config config_;
    float deltaTimeSeconds_{0.1f};
    float elapsedTimeSeconds_{0.0f};
    std::uint64_t tickIndex_{0};

    std::chrono::steady_clock::time_point nextTickDeadline_{};
    bool pacingInitialized_{false};
};

} // namespace sim::core
