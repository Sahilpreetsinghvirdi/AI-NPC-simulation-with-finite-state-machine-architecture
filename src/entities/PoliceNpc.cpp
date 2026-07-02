#include "sim/entities/PoliceNpc.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sim::entities {

namespace {

constexpr float kAttackRange = 10.0f;
constexpr float kAttackCooldown = 1.4f;
constexpr float kAttackDamage = 8.0f;
constexpr float kAcceleration = 28.0f;
constexpr float kFarPursuitDistance = 45.0f;
constexpr float kCloseSpeedMultiplier = 0.45f;

sim::math::Vec2 DirectionFromAngle(const float radians)
{
    return {std::cos(radians), std::sin(radians)};
}

float MoveTowards(const float current, const float target, const float maxDelta)
{
    if (current < target) {
        return std::min(target, current + maxDelta);
    }
    return std::max(target, current - maxDelta);
}

} // namespace

PoliceNpc::PoliceNpc() = default;

PoliceNpc::PoliceNpc(const sim::math::Vec2 startPosition)
    : position_(startPosition)
    , currentSpeed_(PoliceNpc::kBaseSpeed)
{
}

const sim::math::Vec2& PoliceNpc::GetPosition() const
{
    return position_;
}

float PoliceNpc::GetHealth() const
{
    return health_;
}

bool PoliceNpc::IsAlive() const
{
    return health_ > 0.0f;
}

NpcState PoliceNpc::GetState() const
{
    return stateMachine_.GetState();
}

NpcAction PoliceNpc::GetLastAction() const
{
    return lastAction_;
}

float PoliceNpc::GetCurrentSpeed() const
{
    return currentSpeed_;
}

float PoliceNpc::GetSpeedMultiplier() const
{
    return currentSpeed_ / kBaseSpeed;
}

float PoliceNpc::GetEffectiveMaxSpeed() const
{
    return effectiveMaxSpeed_;
}

const sim::math::Vec2& PoliceNpc::GetCurrentTarget() const
{
    return currentTarget_;
}

bool PoliceNpc::Update(const float deltaTime, Player& player)
{
    return Update(deltaTime, player, player.GetPosition(), {}, true);
}

bool PoliceNpc::Update(const float deltaTime,
                       Player& player,
                       const sim::math::Vec2 interceptTarget,
                       const sim::math::Vec2 separationForce,
                       const bool canDamagePlayer)
{
    if (!IsAlive()) {
        lastAction_ = NpcAction::Wait;
        return false;
    }

    attackCooldown_ = std::max(0.0f, attackCooldown_ - deltaTime);
    currentTarget_ = interceptTarget;

    const sim::ai::NpcDecision decision = stateMachine_.Update(BuildAiContext(player), deltaTime);
    UpdateSpeed(deltaTime, player);

    const float distance = sim::math::Distance(position_, player.GetPosition());
    if (player.IsAlive() && player.GetWantedLevel() > 0 && distance <= kAttackRange && attackCooldown_ <= 0.0f) {
        return TryAttack(player, deltaTime, canDamagePlayer);
    }

    lastAction_ = decision.action;
    if (decision.action == NpcAction::MoveTowardPlayer) {
        const sim::math::Vec2 toTarget = (currentTarget_ - position_).Normalized();
        MoveBy(((toTarget * currentSpeed_) + separationForce) * deltaTime);
    } else {
        ApplyAction(decision.action, deltaTime, player);
    }

    return false;
}

sim::ai::Observation PoliceNpc::GetObservation(const Player& player) const
{
    const float distance = sim::math::Distance(position_, player.GetPosition());

    sim::ai::Observation observation;
    observation.features = {
        position_.x,
        position_.y,
        player.GetPosition().x,
        player.GetPosition().y,
        distance,
        static_cast<float>(player.GetWantedLevel()),
        health_,
        static_cast<float>(stateMachine_.GetState())
    };

    return observation;
}

void PoliceNpc::ApplyAction(const NpcAction action, const float deltaTime, const Player& player)
{
    // TODO: Route all locomotion through a learned policy once RL is integrated.
    switch (action) {
    case NpcAction::Wait:
        break;

    case NpcAction::PatrolStep: {
        patrolHeadingRadians_ += 0.05f;
        MoveBy(DirectionFromAngle(patrolHeadingRadians_) * (currentSpeed_ * deltaTime));
        break;
    }

    case NpcAction::MoveTowardPlayer: {
        const sim::math::Vec2 toPlayer = (player.GetPosition() - position_).Normalized();
        MoveBy(toPlayer * (currentSpeed_ * deltaTime));
        break;
    }

    case NpcAction::Attack:
        break;

    case NpcAction::None:
    default:
        break;
    }
}

float PoliceNpc::CalculateReward(const Player& player, const NpcAction /*action*/) const
{
    if (player.GetWantedLevel() <= 0) {
        return 0.0f;
    }

    const float distance = sim::math::Distance(position_, player.GetPosition());
    return std::max(0.0f, 100.0f - distance);
}

std::string PoliceNpc::ToString() const
{
    std::ostringstream stream;
    stream << "Police{pos=" << position_
           << ", health=" << health_
           << ", speed=" << currentSpeed_
           << ", state=" << sim::entities::ToString(stateMachine_.GetState())
           << '}';
    return stream.str();
}

void PoliceNpc::ApplyDamage(const float amount)
{
    if (amount <= 0.0f) {
        return;
    }

    health_ = std::max(0.0f, health_ - amount);
}

sim::ai::NpcAiContext PoliceNpc::BuildAiContext(const Player& player) const
{
    return {
        .npcPosition = position_,
        .playerPosition = player.GetPosition(),
        .playerWantedLevel = player.GetWantedLevel(),
        .playerAlive = player.IsAlive()
    };
}

void PoliceNpc::UpdateSpeed(const float deltaTime, const Player& player)
{
    const float distance = sim::math::Distance(position_, player.GetPosition());
    float maxSpeed = kBaseSpeed;

    if (player.GetWantedLevel() <= 3) {
        maxSpeed = kBaseSpeed;
    } else {
        const int boostedStars = player.GetWantedLevel() - 3;
        float speedMultiplier = 1.0f + (kSpeedIncreasePerStar * static_cast<float>(boostedStars));
        if (player.GetWantedLevel() >= Player::kMaxWantedLevel) {
            speedMultiplier = std::max(speedMultiplier, kMaxPoliceSpeed / kBaseSpeed);
        }

        maxSpeed = std::min(kBaseSpeed * speedMultiplier, kMaxPoliceSpeed);
    }

    float distanceMultiplier = 1.0f;
    if (distance <= kAttackRange) {
        distanceMultiplier = kCloseSpeedMultiplier;
    } else if (distance < kFarPursuitDistance) {
        const float t = (distance - kAttackRange) / (kFarPursuitDistance - kAttackRange);
        distanceMultiplier = kCloseSpeedMultiplier + ((1.0f - kCloseSpeedMultiplier) * t);
    }

    effectiveMaxSpeed_ = maxSpeed;
    currentSpeed_ = MoveTowards(currentSpeed_, maxSpeed * distanceMultiplier, kAcceleration * deltaTime);
}

bool PoliceNpc::TryAttack(Player& player, const float /*deltaTime*/, const bool canDamagePlayer)
{
    if (!canDamagePlayer || !player.IsAlive() || player.GetWantedLevel() <= 0) {
        return false;
    }

    const float distance = sim::math::Distance(position_, player.GetPosition());
    if (distance > kAttackRange || attackCooldown_ > 0.0f) {
        return false;
    }

    player.ApplyDamage(kAttackDamage);
    attackCooldown_ = kAttackCooldown;
    lastAction_ = NpcAction::Attack;
    return true;
}

void PoliceNpc::MoveBy(const sim::math::Vec2& offset)
{
    position_ += offset;
}

} // namespace sim::entities
