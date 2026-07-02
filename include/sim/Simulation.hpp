#pragma once

#include "sim/core/Logger.hpp"
#include "sim/core/SimulationTimer.hpp"
#include "sim/entities/Player.hpp"
#include "sim/entities/PoliceManager.hpp"
#include "sim/entities/PoliceNpc.hpp"
#include "sim/math/Vec2.hpp"

#include <cstdint>
#include <string>

namespace sim {

// Orchestrates entities, timing, and logging for the AI simulation sandbox.
class Simulation {
public:
    struct Config {
        std::uint64_t maxTicks{0}; // 0 = run until the window closes
        float tickRateHz{10.0f};
        bool realTimePacing{false};
    };

    Simulation(sim::core::Logger& logger);
    Simulation(sim::core::Logger& logger, Config config);

    void Run();
    void Tick();
    void UpdateRealtime(float frameDeltaSeconds);

    [[nodiscard]] bool IsFinished() const;
    [[nodiscard]] bool IsPaused() const;
    [[nodiscard]] bool IsPlayerDead() const;
    [[nodiscard]] float GetDeathPauseRemaining() const;
    [[nodiscard]] const std::string& GetLastEventMessage() const;
    [[nodiscard]] sim::math::Vec2 GetHospitalPosition() const;
    [[nodiscard]] float GetPlayerCurrentSpeed() const;
    [[nodiscard]] float GetPursuitTimerSeconds() const;
    [[nodiscard]] float GetPursuitFailureSeconds() const;

    [[nodiscard]] const sim::entities::Player& GetPlayer() const { return player_; }
    [[nodiscard]] const sim::entities::PoliceManager& GetPoliceManager() const { return policeManager_; }
    [[nodiscard]] const sim::entities::PoliceNpc* GetPrimaryPoliceNpc() const { return policeManager_.GetPrimaryPolice(); }
    [[nodiscard]] const sim::core::SimulationTimer& GetTimer() const { return timer_; }
    [[nodiscard]] sim::core::SimulationTimer& GetTimer() { return timer_; }

private:
    void UpdatePlayer(float deltaTime);
    void UpdatePolice(float deltaTime);
    void UpdatePursuitEscalation(float deltaTime, float playerHealthBeforePoliceUpdate);
    void TryPlayerAttackPolice(float deltaTime);
    [[nodiscard]] float CalculatePlayerTargetSpeed() const;
    void BeginDeathSequence();
    void RespawnPlayer();
    void RecordEvent(std::string message);

    sim::core::Logger& logger_;
    sim::core::SimulationTimer timer_;
    sim::entities::Player player_;
    sim::entities::PoliceManager policeManager_;
    Config config_;
    sim::math::Vec2 hospitalPosition_{15.0f, 15.0f};
    float playerHeadingRadians_{0.0f};
    float playerCurrentSpeed_{0.0f};
    float playerAttackCooldown_{0.0f};
    float pursuitTimerSeconds_{0.0f};
    float deathPauseRemaining_{0.0f};
    std::string lastEventMessage_;
};

} // namespace sim
