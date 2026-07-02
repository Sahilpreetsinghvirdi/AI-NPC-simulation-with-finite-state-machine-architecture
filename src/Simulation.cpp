#include "sim/Simulation.hpp"

#include <cmath>

namespace sim {

namespace {

constexpr float kPlayerMoveSpeed = 8.0f;
constexpr float kDeathPauseDuration = 3.0f;

} // namespace

Simulation::Simulation(sim::core::Logger& logger)
    : Simulation(logger, Config{})
{
}

Simulation::Simulation(sim::core::Logger& logger, const Config config)
    : logger_(logger)
    , timer_({.tickRateHz = config.tickRateHz, .realTimePacing = config.realTimePacing})
    , player_({20.0f, 20.0f})
    , police_({80.0f, 80.0f})
    , config_(config)
{
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
    UpdatePlayer(deltaTime);
    UpdatePolice(deltaTime);

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

void Simulation::UpdatePlayer(const float deltaTime)
{
    if (player_.IsDead()) {
        return;
    }

    playerHeadingRadians_ += 0.08f;
    const sim::math::Vec2 direction{std::cos(playerHeadingRadians_), std::sin(playerHeadingRadians_)};
    player_.Translate(direction * (kPlayerMoveSpeed * deltaTime));

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
    police_.Update(deltaTime, player_);
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
    player_.DecreaseWantedLevel(1);
    deathPauseRemaining_ = 0.0f;
    RecordEvent("Player respawned at hospital. Wanted level reduced.");
}

void Simulation::RecordEvent(std::string message)
{
    lastEventMessage_ = std::move(message);
    logger_.Warn(lastEventMessage_);
}

} // namespace sim
