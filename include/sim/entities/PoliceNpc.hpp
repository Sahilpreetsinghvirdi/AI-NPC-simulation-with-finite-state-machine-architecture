#pragma once

#include "sim/ai/Observation.hpp"
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
    static constexpr float kMaxHunger = 100.0f;

    PoliceNpc();
    explicit PoliceNpc(sim::math::Vec2 startPosition);

    [[nodiscard]] const sim::math::Vec2& GetPosition() const;
    [[nodiscard]] float GetHealth() const;
    [[nodiscard]] float GetHunger() const;
    [[nodiscard]] float GetMoney() const;
    [[nodiscard]] NpcState GetState() const;
    [[nodiscard]] NpcAction GetLastAction() const;
    [[nodiscard]] float GetCurrentSpeed() const;

    void Update(float deltaTime, Player& player);

    // RL interface — stable contract for future policy integration.
    [[nodiscard]] sim::ai::Observation GetObservation(const Player& player) const;
    void ApplyAction(NpcAction action, float deltaTime, const Player& player);
    [[nodiscard]] float CalculateReward(const Player& player, NpcAction action) const;

    [[nodiscard]] std::string ToString() const;

private:
    void DecideState(const Player& player);
    NpcAction SelectRuleBasedAction(const Player& player) const;
    void ApplyHungerDecay(float deltaTime);
    void UpdateSpeed(float deltaTime, const Player& player);
    void TryAttack(Player& player, float deltaTime);
    [[nodiscard]] float GetTargetSpeedMultiplier(float distanceToPlayer, const Player& player) const;
    void MoveBy(const sim::math::Vec2& offset);

    sim::math::Vec2 position_;
    float health_{kMaxHealth};
    float hunger_{kMaxHunger};
    float money_{250.0f};
    NpcState state_{NpcState::Idle};
    NpcAction lastAction_{NpcAction::None};
    float patrolHeadingRadians_{0.0f};
    float currentSpeed_{0.0f};
    float attackCooldown_{0.0f};
};

} // namespace sim::entities
