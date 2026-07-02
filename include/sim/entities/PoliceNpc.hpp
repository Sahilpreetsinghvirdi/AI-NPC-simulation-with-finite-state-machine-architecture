#pragma once

#include "sim/ai/Observation.hpp"
#include "sim/ai/NpcFiniteStateMachine.hpp"
#include "sim/entities/NpcAction.hpp"
#include "sim/entities/NpcState.hpp"
#include "sim/entities/Player.hpp"
#include "sim/math/Vec2.hpp"

#include <string>

namespace sim::entities {

// Police NPC with an RL-ready interface. Rule-based logic drives behavior until training lands.
class PoliceNpc {
public:
    static constexpr float kMaxHealth = 100.0f;
    static constexpr float kBaseSpeed = 8.0f;
    static constexpr float kSpeedIncreasePerStar = 0.10f;
    static constexpr float kMaxPoliceSpeed = 19.2f;

    PoliceNpc();
    explicit PoliceNpc(sim::math::Vec2 startPosition);

    [[nodiscard]] const sim::math::Vec2& GetPosition() const;
    [[nodiscard]] float GetHealth() const;
    [[nodiscard]] bool IsAlive() const;
    [[nodiscard]] NpcState GetState() const;
    [[nodiscard]] NpcAction GetLastAction() const;
    [[nodiscard]] float GetCurrentSpeed() const;
    [[nodiscard]] float GetSpeedMultiplier() const;
    [[nodiscard]] float GetEffectiveMaxSpeed() const;

    void Update(float deltaTime, Player& player);
    void ApplyDamage(float amount);

    // RL interface — stable contract for future policy integration.
    [[nodiscard]] sim::ai::Observation GetObservation(const Player& player) const;
    void ApplyAction(NpcAction action, float deltaTime, const Player& player);
    [[nodiscard]] float CalculateReward(const Player& player, NpcAction action) const;

    [[nodiscard]] std::string ToString() const;

private:
    [[nodiscard]] sim::ai::NpcAiContext BuildAiContext(const Player& player) const;
    void UpdateSpeed(float deltaTime, const Player& player);
    void TryAttack(Player& player, float deltaTime);
    void MoveBy(const sim::math::Vec2& offset);

    sim::math::Vec2 position_;
    float health_{kMaxHealth};
    sim::ai::NpcFiniteStateMachine stateMachine_;
    NpcAction lastAction_{NpcAction::None};
    float patrolHeadingRadians_{0.0f};
    float currentSpeed_{0.0f};
    float attackCooldown_{0.0f};
};

} // namespace sim::entities
