#pragma once

#include "sim/math/Vec2.hpp"

#include <string>

namespace sim::entities {

enum class PlayerLifeState {
    Alive = 0,
    Dead
};

const char* ToString(PlayerLifeState state);

// Player state tracked by the simulation and observed by NPC AI systems.
class Player {
public:
    static constexpr int kMaxWantedLevel = 5;
    static constexpr float kMaxHealth = 100.0f;

    Player();
    explicit Player(sim::math::Vec2 startPosition, float startHealth = kMaxHealth);

    [[nodiscard]] const sim::math::Vec2& GetPosition() const;
    [[nodiscard]] float GetHealth() const;
    [[nodiscard]] int GetWantedLevel() const;
    [[nodiscard]] PlayerLifeState GetLifeState() const;
    [[nodiscard]] bool IsAlive() const;
    [[nodiscard]] bool IsDead() const;

    void SetPosition(const sim::math::Vec2& position);
    void SetHealth(float health);
    void SetWantedLevel(int wantedLevel);

    void Translate(const sim::math::Vec2& offset);
    void ApplyDamage(float amount);
    void Heal(float amount);
    void IncreaseWantedLevel(int amount = 1);
    void DecreaseWantedLevel(int amount = 1);
    void ClearWantedLevel();
    void MarkDead();
    void Respawn(sim::math::Vec2 position, float health = kMaxHealth);

    [[nodiscard]] std::string ToString() const;

private:
    void ClampHealth();
    void ClampWantedLevel();

    sim::math::Vec2 position_;
    float health_{kMaxHealth};
    int wantedLevel_{0};
    PlayerLifeState lifeState_{PlayerLifeState::Alive};
};

} // namespace sim::entities
