#include "sim/entities/PoliceNpc.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace sim::entities {

namespace {

constexpr float kAttackRange = 10.0f;
constexpr float kAttackCooldown = 0.5f;
constexpr float kAttackDamage = 20.0f;

sim::math::Vec2 DirectionFromAngle(const float radians)
{
    return {std::cos(radians), std::sin(radians)};
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
    return currentSpeed_;
}

void PoliceNpc::Update(const float deltaTime, Player& player)
{
    if (!IsAlive()) {
        lastAction_ = NpcAction::Wait;
        return;
    }

    attackCooldown_ = std::max(0.0f, attackCooldown_ - deltaTime);

    const sim::ai::NpcDecision decision = stateMachine_.Update(BuildAiContext(player), deltaTime);
    UpdateSpeed(deltaTime, player);

    const float distance = sim::math::Distance(position_, player.GetPosition());
    if (player.IsAlive() && player.GetWantedLevel() > 0 && distance <= kAttackRange && attackCooldown_ <= 0.0f) {
        TryAttack(player, deltaTime);
        return;
    }

    lastAction_ = decision.action;
    ApplyAction(decision.action, deltaTime, player);
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

void PoliceNpc::UpdateSpeed(const float /*deltaTime*/, const Player& player)
{
    if (player.GetWantedLevel() <= 3) {
        currentSpeed_ = kBaseSpeed;
        return;
    }

    const int boostedStars = player.GetWantedLevel() - 3;
    float speedMultiplier = 1.0f + (kSpeedIncreasePerStar * static_cast<float>(boostedStars));
    if (player.GetWantedLevel() >= Player::kMaxWantedLevel) {
        speedMultiplier = std::max(speedMultiplier, kMaxPoliceSpeed / kBaseSpeed);
    }

    currentSpeed_ = std::min(kBaseSpeed * speedMultiplier, kMaxPoliceSpeed);
}

void PoliceNpc::TryAttack(Player& player, const float /*deltaTime*/)
{
    if (!player.IsAlive() || player.GetWantedLevel() <= 0) {
        return;
    }

    const float distance = sim::math::Distance(position_, player.GetPosition());
    if (distance > kAttackRange || attackCooldown_ > 0.0f) {
        return;
    }

    player.ApplyDamage(kAttackDamage);
    attackCooldown_ = kAttackCooldown;
    lastAction_ = NpcAction::Attack;
}

void PoliceNpc::MoveBy(const sim::math::Vec2& offset)
{
    position_ += offset;
}

} // namespace sim::entities
