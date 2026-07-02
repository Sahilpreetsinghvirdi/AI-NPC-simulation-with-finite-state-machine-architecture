#pragma once

#include "sim/entities/NpcAction.hpp"
#include "sim/entities/NpcState.hpp"
#include "sim/math/Vec2.hpp"

namespace sim::ai {

struct NpcAiContext {
    sim::math::Vec2 npcPosition;
    sim::math::Vec2 playerPosition;
    int playerWantedLevel{0};
    bool playerAlive{true};
};

struct NpcDecision {
    sim::entities::NpcState state{sim::entities::NpcState::Idle};
    sim::entities::NpcAction action{sim::entities::NpcAction::None};
    bool stateChanged{false};
    float timeInState{0.0f};
};

class NpcFiniteStateMachine {
public:
    explicit NpcFiniteStateMachine(sim::entities::NpcState initialState = sim::entities::NpcState::Idle);

    [[nodiscard]] sim::entities::NpcState GetState() const;
    [[nodiscard]] float GetTimeInState() const;

    void Reset(sim::entities::NpcState state = sim::entities::NpcState::Idle);
    NpcDecision Update(const NpcAiContext& context, float deltaTime);

private:
    [[nodiscard]] sim::entities::NpcState EvaluateState(const NpcAiContext& context) const;
    [[nodiscard]] sim::entities::NpcAction SelectAction(const NpcAiContext& context) const;
    void TransitionTo(sim::entities::NpcState nextState);

    sim::entities::NpcState state_{sim::entities::NpcState::Idle};
    float timeInState_{0.0f};
};

} // namespace sim::ai
