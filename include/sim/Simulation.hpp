#pragma once

#include "sim/core/Logger.hpp"
#include "sim/core/SimulationTimer.hpp"
#include "sim/entities/Player.hpp"
#include "sim/entities/PoliceManager.hpp"
#include "sim/entities/PoliceNpc.hpp"
#include "sim/math/Vec2.hpp"
#include "sim/ai/PersistentLearningPolicy.hpp"
#include "sim/rl/CheckpointManager.hpp"
#include "sim/rl/EpisodeRecorder.hpp"
#include "sim/rl/RlTypes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sim {

// Orchestrates entities, timing, and logging for the AI simulation sandbox.
class Simulation {
public:
    struct PerceivedBoundary {
        sim::math::Vec2 start;
        sim::math::Vec2 end;
        sim::math::Vec2 normal;
        float lastSeenSeconds{0.0f};
        bool visible{false};
        bool remembered{false};
    };

    struct EscapeExperience {
        sim::math::Vec2 position;
        sim::math::Vec2 escapeDirection;
        std::size_t nearbyPoliceCount{0};
        float timeSurvived{0.0f};
        bool success{false};
    };

    struct Config {
        std::uint64_t maxTicks{0}; // 0 = run until the window closes
        float tickRateHz{10.0f};
        bool realTimePacing{false};
        sim::rl::RunMode rlMode{sim::rl::RunMode::Training};
        std::size_t maxStoredRlTransitions{4096};
        std::filesystem::path checkpointPath{};
        std::uint64_t checkpointAutosaveIntervalSteps{50};
    };

    Simulation(sim::core::Logger& logger);
    Simulation(sim::core::Logger& logger, Config config);
    ~Simulation();

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
    [[nodiscard]] sim::math::Vec2 GetPlayerVelocity() const { return playerVelocity_; }
    [[nodiscard]] sim::math::Vec2 GetPlayerCurrentHeading() const { return playerCurrentHeading_; }
    [[nodiscard]] sim::math::Vec2 GetPlayerDesiredHeading() const { return playerDesiredHeading_; }
    [[nodiscard]] sim::math::Vec2 GetPlayerSteeringVector() const { return playerSteeringVector_; }
    [[nodiscard]] sim::math::Vec2 GetPlayerEscapeVector() const { return playerEscapeVector_; }
    [[nodiscard]] float GetPlayerPerceptionRadius() const;
    [[nodiscard]] const std::array<PerceivedBoundary, 4>& GetPlayerKnownBoundaries() const { return playerKnownBoundaries_; }
    [[nodiscard]] std::size_t GetPlayerPerceivedObstacleCount() const;
    [[nodiscard]] float GetPlayerEscapeScore() const { return playerEscapeScore_; }
    [[nodiscard]] float GetPlayerDangerLevel() const { return playerDangerLevel_; }
    [[nodiscard]] const std::vector<EscapeExperience>& GetPlayerExperiences() const { return playerExperiences_; }
    [[nodiscard]] std::size_t GetPlayerSuccessfulEscapeCount() const;
    [[nodiscard]] std::size_t GetPlayerFailedEscapeCount() const;
    [[nodiscard]] sim::math::Vec2 GetPlayerPreferredEscapeDirection() const { return playerPreferredEscapeDirection_; }
    [[nodiscard]] float GetPursuitTimerSeconds() const;
    [[nodiscard]] float GetPursuitFailureSeconds() const;

    [[nodiscard]] const sim::entities::Player& GetPlayer() const { return player_; }
    [[nodiscard]] const sim::entities::PoliceManager& GetPoliceManager() const { return policeManager_; }
    [[nodiscard]] const sim::entities::PoliceNpc* GetPrimaryPoliceNpc() const { return policeManager_.GetPrimaryPolice(); }
    [[nodiscard]] const sim::core::SimulationTimer& GetTimer() const { return timer_; }
    [[nodiscard]] sim::core::SimulationTimer& GetTimer() { return timer_; }
    [[nodiscard]] const sim::rl::EpisodeRecorder& GetRlEpisodeRecorder() const { return rlEpisodeRecorder_; }
    [[nodiscard]] const sim::ai::PersistentLearningPolicy& GetLearningPolicy() const { return learningPolicy_; }
    [[nodiscard]] sim::ai::AiDebugSnapshot BuildAiDebugSnapshot(float fps) const;

private:
    void UpdatePlayer(float deltaTime);
    void UpdatePlayerPerception(float deltaTime);
    void UpdatePlayerExperienceMemory(float deltaTime, float bestFreeSpace);
    void UpdatePolice(float deltaTime);
    void RecordPoliceRlTransitions(const std::vector<sim::ai::Observation>& observationsBeforeUpdate);
    void LoadLearningCheckpoint();
    void SaveLearningCheckpoint(std::string_view reason);
    void MaybeAutosaveLearningCheckpoint();
    void UpdatePursuitEscalation(float deltaTime, float playerHealthBeforePoliceUpdate);
    void TryPlayerAttackPolice(float deltaTime);
    [[nodiscard]] float CalculatePlayerTargetSpeed() const;
    void BeginDeathSequence();
    void RespawnPlayer();
    void RecordEscapeExperience(bool success, float timeSurvived);
    [[nodiscard]] float CalculateExperienceBias(sim::math::Vec2 position,
                                                sim::math::Vec2 direction,
                                                std::size_t nearbyPoliceCount) const;
    [[nodiscard]] std::size_t CountNearbyPolice(float radius) const;
    void RecordEvent(std::string message);

    sim::core::Logger& logger_;
    sim::core::SimulationTimer timer_;
    sim::entities::Player player_;
    sim::entities::PoliceManager policeManager_;
    Config config_;
    sim::rl::EpisodeRecorder rlEpisodeRecorder_;
    sim::ai::PersistentLearningPolicy learningPolicy_;
    std::unique_ptr<sim::rl::CheckpointManager> checkpointManager_;
    std::uint64_t lastCheckpointStep_{0};
    sim::math::Vec2 hospitalPosition_{15.0f, 15.0f};
    sim::math::Vec2 playerVelocity_;
    sim::math::Vec2 playerCurrentHeading_{1.0f, 0.0f};
    sim::math::Vec2 playerDesiredHeading_{1.0f, 0.0f};
    sim::math::Vec2 playerSteeringVector_;
    sim::math::Vec2 playerEscapeVector_{1.0f, 0.0f};
    std::array<PerceivedBoundary, 4> playerKnownBoundaries_;
    std::vector<EscapeExperience> playerExperiences_;
    sim::math::Vec2 playerPreferredEscapeDirection_{1.0f, 0.0f};
    float playerEscapeScore_{0.0f};
    float playerDangerLevel_{0.0f};
    float playerTrapTimer_{0.0f};
    float playerExperienceSurvivalTimer_{0.0f};
    float playerSuccessRecordCooldown_{0.0f};
    float playerCurrentSpeed_{0.0f};
    float playerAttackCooldown_{0.0f};
    float pursuitTimerSeconds_{0.0f};
    float deathPauseRemaining_{0.0f};
    std::string lastEventMessage_;
};

} // namespace sim
