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
    [[nodiscard]] const sim::math::Vec2& GetCurrentTarget() const;
    [[nodiscard]] const sim::math::Vec2& GetCurrentHeading() const;
    [[nodiscard]] const sim::math::Vec2& GetDesiredHeading() const;
    [[nodiscard]] const sim::math::Vec2& GetSteeringVector() const;
    [[nodiscard]] const sim::math::Vec2& GetPursuitVector() const;

    bool Update(float deltaTime, Player& player);
    bool Update(float deltaTime,
                Player& player,
                sim::math::Vec2 interceptTarget,
                sim::math::Vec2 separationForce,
                bool canDamagePlayer);
    void ApplyDamage(float amount);

    // RL interface — stable contract for future policy integration.
    [[nodiscard]] sim::ai::Observation GetObservation(const Player& player) const;
    void ApplyAction(NpcAction action, float deltaTime, const Player& player);
    [[nodiscard]] float CalculateReward(const Player& player, NpcAction action) const;

    [[nodiscard]] std::string ToString() const;

private:
    [[nodiscard]] sim::ai::NpcAiContext BuildAiContext(const Player& player) const;
    void UpdateSpeed(float deltaTime, const Player& player);
    [[nodiscard]] bool TryAttack(Player& player, float deltaTime, bool canDamagePlayer);
    void MoveBy(const sim::math::Vec2& offset);

    sim::math::Vec2 position_;
    sim::math::Vec2 currentTarget_;
    sim::math::Vec2 currentHeading_{1.0f, 0.0f};
    sim::math::Vec2 desiredHeading_{1.0f, 0.0f};
    sim::math::Vec2 steeringVector_;
    sim::math::Vec2 pursuitVector_;
    float health_{kMaxHealth};
    sim::ai::NpcFiniteStateMachine stateMachine_;
    NpcAction lastAction_{NpcAction::None};
    float patrolHeadingRadians_{0.0f};
    float currentSpeed_{0.0f};
    float effectiveMaxSpeed_{kBaseSpeed};
    float attackCooldown_{0.0f};
};

} // namespace sim::entities
