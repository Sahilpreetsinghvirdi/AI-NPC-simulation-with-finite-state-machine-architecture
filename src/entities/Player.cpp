#include "sim/entities/Player.hpp"

#include <algorithm>
#include <sstream>

namespace sim::entities {

Player::Player() = default;

Player::Player(const sim::math::Vec2 startPosition, const float startHealth)
    : position_(startPosition)
    , health_(startHealth)
{
    ClampHealth();
}

const sim::math::Vec2& Player::GetPosition() const
{
    return position_;
}

float Player::GetHealth() const
{
    return health_;
}

int Player::GetWantedLevel() const
{
    return wantedLevel_;
}

PlayerLifeState Player::GetLifeState() const
{
    return lifeState_;
}

bool Player::IsAlive() const
{
    return lifeState_ == PlayerLifeState::Alive && health_ > 0.0f;
}

bool Player::IsDead() const
{
    return lifeState_ == PlayerLifeState::Dead;
}

void Player::SetPosition(const sim::math::Vec2& position)
{
    position_ = position;
}

void Player::SetHealth(const float health)
{
    health_ = health;
    ClampHealth();
}

void Player::SetWantedLevel(const int wantedLevel)
{
    wantedLevel_ = wantedLevel;
    ClampWantedLevel();
}

void Player::Translate(const sim::math::Vec2& offset)
{
    if (IsDead()) {
        return;
    }

    position_ += offset;
}

void Player::ApplyDamage(const float amount)
{
    if (amount <= 0.0f) {
        return;
    }

    health_ -= amount;
    ClampHealth();
}

void Player::Heal(const float amount)
{
    if (amount <= 0.0f) {
        return;
    }

    health_ += amount;
    ClampHealth();
}

void Player::IncreaseWantedLevel(const int amount)
{
    if (amount <= 0) {
        return;
    }

    wantedLevel_ += amount;
    ClampWantedLevel();
}

void Player::DecreaseWantedLevel(const int amount)
{
    if (amount <= 0) {
        return;
    }

    wantedLevel_ -= amount;
    ClampWantedLevel();
}

void Player::ClearWantedLevel()
{
    wantedLevel_ = 0;
}

void Player::MarkDead()
{
    lifeState_ = PlayerLifeState::Dead;
    health_ = 0.0f;
}

void Player::Respawn(const sim::math::Vec2 position, const float health)
{
    lifeState_ = PlayerLifeState::Alive;
    position_ = position;
    health_ = health;
    ClampHealth();
}

std::string Player::ToString() const
{
    std::ostringstream stream;
    stream << "Player{pos=" << position_
           << ", health=" << health_
           << ", wanted=" << wantedLevel_
           << ", state=" << sim::entities::ToString(lifeState_)
           << '}';
    return stream.str();
}

void Player::ClampHealth()
{
    health_ = std::clamp(health_, 0.0f, kMaxHealth);
}

void Player::ClampWantedLevel()
{
    wantedLevel_ = std::clamp(wantedLevel_, 0, kMaxWantedLevel);
}

} // namespace sim::entities
