#include "sim/Simulation.hpp"

#include "sim/entities/NpcState.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace sim {

namespace {

constexpr float kPlayerBaseSpeed = 6.0f;
constexpr float kPlayerMaxSpeed = 18.0f;
constexpr float kPlayerAcceleration = 22.0f;
constexpr float kPlayerAccelerationDistance = 40.0f;
constexpr float kPlayerAttackRange = 8.0f;
constexpr float kPlayerAttackCooldown = 0.8f;
constexpr float kPlayerAttackDamage = 8.0f;
constexpr float kPursuitFailureDuration = 10.0f;
constexpr float kDeathPauseDuration = 5.0f;
constexpr int kRespawnWantedLevel = 3;
constexpr float kGoalArrivalDistance = 4.0f;
constexpr std::array<sim::math::Vec2, 5> kPlayerGoals{
    sim::math::Vec2{100.0f, 25.0f},
    sim::math::Vec2{102.0f, 98.0f},
    sim::math::Vec2{28.0f, 104.0f},
    sim::math::Vec2{18.0f, 38.0f},
    sim::math::Vec2{78.0f, 58.0f}
};

float MoveTowards(const float current, const float target, const float maxDelta)
{
    if (current < target) {
        return std::min(target, current + maxDelta);
    }
    return std::max(target, current - maxDelta);
}

} // namespace

Simulation::Simulation(sim::core::Logger& logger)
    : Simulation(logger, Config{})
{
}

Simulation::Simulation(sim::core::Logger& logger, const Config config)
    : logger_(logger)
    , timer_({.tickRateHz = config.tickRateHz, .realTimePacing = config.realTimePacing})
    , player_({20.0f, 20.0f})
    , config_(config)
{
    player_.SetWantedLevel(1);
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
}

void Simulation::Run()
{
    logger_.Info("Simulation started (headless).");

    while (!IsFinished()) {
        Tick();
    }

    logger_.Info("Simulation finished at t=" + std::to_string(timer_.GetElapsedTime()) + "s");
}

void Simulation::UpdateRealtime(const float frameDeltaSeconds)
{
    if (deathPauseRemaining_ <= 0.0f) {
        return;
    }

    deathPauseRemaining_ -= frameDeltaSeconds;
    if (deathPauseRemaining_ <= 0.0f) {
        RespawnPlayer();
    }
}

void Simulation::Tick()
{
    if (IsPaused()) {
        return;
    }

    timer_.Advance();

    const float deltaTime = timer_.GetDeltaTime();
    playerAttackCooldown_ = std::max(0.0f, playerAttackCooldown_ - deltaTime);
    UpdatePlayer(deltaTime);
    const float playerHealthBeforePoliceUpdate = player_.GetHealth();
    UpdatePolice(deltaTime);
    UpdatePursuitEscalation(deltaTime, playerHealthBeforePoliceUpdate);
    TryPlayerAttackPolice(deltaTime);

    if (player_.GetHealth() <= 0.0f && !player_.IsDead()) {
        BeginDeathSequence();
    }
}

bool Simulation::IsFinished() const
{
    return config_.maxTicks > 0 && timer_.GetTickIndex() >= config_.maxTicks;
}

bool Simulation::IsPaused() const
{
    return deathPauseRemaining_ > 0.0f;
}

bool Simulation::IsPlayerDead() const
{
    return player_.IsDead();
}

float Simulation::GetDeathPauseRemaining() const
{
    return deathPauseRemaining_;
}

const std::string& Simulation::GetLastEventMessage() const
{
    return lastEventMessage_;
}

sim::math::Vec2 Simulation::GetHospitalPosition() const
{
    return hospitalPosition_;
}

float Simulation::GetPlayerCurrentSpeed() const
{
    return playerCurrentSpeed_;
}

float Simulation::GetPursuitTimerSeconds() const
{
    return pursuitTimerSeconds_;
}

float Simulation::GetPursuitFailureSeconds() const
{
    return kPursuitFailureDuration;
}

void Simulation::UpdatePlayer(const float deltaTime)
{
    if (player_.IsDead()) {
        playerCurrentSpeed_ = 0.0f;
        playerVelocity_ = {};
        return;
    }

    playerCurrentSpeed_ = MoveTowards(playerCurrentSpeed_, CalculatePlayerTargetSpeed(), kPlayerAcceleration * deltaTime);

    const sim::math::Vec2 goal = kPlayerGoals[playerGoalIndex_];
    if (sim::math::Distance(player_.GetPosition(), goal) <= kGoalArrivalDistance) {
        playerGoalIndex_ = (playerGoalIndex_ + 1) % kPlayerGoals.size();
    }

    const sim::math::Vec2 direction = (kPlayerGoals[playerGoalIndex_] - player_.GetPosition()).Normalized();
    playerVelocity_ = direction * playerCurrentSpeed_;
    player_.Translate(playerVelocity_ * deltaTime);

    const auto tick = timer_.GetTickIndex();
    if (tick == 5) {
        player_.IncreaseWantedLevel(2);
        RecordEvent("Player committed a crime — wanted level increased.");
    } else if (tick == 15) {
        player_.IncreaseWantedLevel(1);
        RecordEvent("Player committed another offense — wanted level increased.");
    }
}

void Simulation::UpdatePolice(const float deltaTime)
{
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
    policeManager_.UpdateAll(deltaTime, player_, playerVelocity_);
}

void Simulation::UpdatePursuitEscalation(const float deltaTime, const float playerHealthBeforePoliceUpdate)
{
    const bool policeDamagedPlayer = player_.GetHealth() < playerHealthBeforePoliceUpdate;
    if (player_.IsDead() || policeDamagedPlayer || !policeManager_.AnyInState(sim::entities::NpcState::Pursue)) {
        pursuitTimerSeconds_ = 0.0f;
        return;
    }

    pursuitTimerSeconds_ += deltaTime;
    if (pursuitTimerSeconds_ < kPursuitFailureDuration) {
        return;
    }

    pursuitTimerSeconds_ = 0.0f;

    if (player_.GetWantedLevel() < sim::entities::Player::kMaxWantedLevel) {
        player_.IncreaseWantedLevel(1);
        policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
        RecordEvent("Pursuit failed for 10s. Wanted level increased.");
    } else {
        RecordEvent("Pursuit failed for 10s. Wanted level is already maxed.");
    }
}

void Simulation::TryPlayerAttackPolice(const float /*deltaTime*/)
{
    if (!player_.IsAlive() || playerAttackCooldown_ > 0.0f) {
        return;
    }

    sim::entities::PoliceNpc* nearestPolice = policeManager_.GetNearestPolice(player_.GetPosition());
    if (nearestPolice == nullptr) {
        return;
    }

    const float distance = sim::math::Distance(player_.GetPosition(), nearestPolice->GetPosition());
    if (distance > kPlayerAttackRange) {
        return;
    }

    nearestPolice->ApplyDamage(kPlayerAttackDamage);
    playerAttackCooldown_ = kPlayerAttackCooldown;
    RecordEvent("Player counterattacked police.");
}

float Simulation::CalculatePlayerTargetSpeed() const
{
    const sim::entities::PoliceNpc* nearestPolice = policeManager_.GetNearestPolice(player_.GetPosition());
    if (nearestPolice == nullptr) {
        return kPlayerBaseSpeed;
    }

    const float distance = sim::math::Distance(player_.GetPosition(), nearestPolice->GetPosition());
    if (distance >= kPlayerAccelerationDistance) {
        return kPlayerBaseSpeed;
    }

    const float closeness = 1.0f - (distance / kPlayerAccelerationDistance);
    return kPlayerBaseSpeed + ((kPlayerMaxSpeed - kPlayerBaseSpeed) * closeness);
}

void Simulation::BeginDeathSequence()
{
    player_.MarkDead();
    deathPauseRemaining_ = kDeathPauseDuration;
    RecordEvent("PLAYER DIED");
}

void Simulation::RespawnPlayer()
{
    player_.Respawn(hospitalPosition_, sim::entities::Player::kMaxHealth);
    player_.SetWantedLevel(kRespawnWantedLevel);
    policeManager_.SyncToWantedLevel(player_.GetWantedLevel(), player_.GetPosition());
    deathPauseRemaining_ = 0.0f;
    RecordEvent("Player respawned at hospital. Wanted level restored to 3 stars.");
}

void Simulation::RecordEvent(std::string message)
{
    lastEventMessage_ = std::move(message);
    logger_.Warn(lastEventMessage_);
}

} // namespace sim
