#include "sim/ai/NpcFiniteStateMachine.hpp"

#include "sim/math/Vec2.hpp"

#include <algorithm>

namespace sim::ai {

namespace {

constexpr float kPursueDistanceThreshold = 80.0f;

} // namespace

NpcFiniteStateMachine::NpcFiniteStateMachine(const sim::entities::NpcState initialState)
    : state_(initialState)
{
}

sim::entities::NpcState NpcFiniteStateMachine::GetState() const
{
    return state_;
}

float NpcFiniteStateMachine::GetTimeInState() const
{
    return timeInState_;
}

void NpcFiniteStateMachine::Reset(const sim::entities::NpcState state)
{
    state_ = state;
    timeInState_ = 0.0f;
}

NpcDecision NpcFiniteStateMachine::Update(const NpcAiContext& context, const float deltaTime)
{
    const sim::entities::NpcState previousState = state_;
    const sim::entities::NpcState nextState = EvaluateState(context);

    if (nextState != state_) {
        TransitionTo(nextState);
    } else {
        timeInState_ += std::max(0.0f, deltaTime);
    }

    return {
        .state = state_,
        .action = SelectAction(context),
        .stateChanged = previousState != state_,
        .timeInState = timeInState_
    };
}

sim::entities::NpcState NpcFiniteStateMachine::EvaluateState(const NpcAiContext& context) const
{
    if (!context.playerAlive) {
        return sim::entities::NpcState::Patrol;
    }

    if (context.playerWantedLevel <= 0) {
        return sim::entities::NpcState::Patrol;
    }

    const float distanceToPlayer = sim::math::Distance(context.npcPosition, context.playerPosition);
    if (distanceToPlayer <= kPursueDistanceThreshold) {
        return sim::entities::NpcState::Pursue;
    }

    return sim::entities::NpcState::Investigate;
}

sim::entities::NpcAction NpcFiniteStateMachine::SelectAction(const NpcAiContext& context) const
{
    switch (state_) {
    case sim::entities::NpcState::Pursue:
        return sim::entities::NpcAction::MoveTowardPlayer;
    case sim::entities::NpcState::Investigate:
        return (context.playerWantedLevel > 0) ? sim::entities::NpcAction::MoveTowardPlayer
                                               : sim::entities::NpcAction::PatrolStep;
    case sim::entities::NpcState::Patrol:
        return sim::entities::NpcAction::PatrolStep;
    case sim::entities::NpcState::Idle:
    default:
        return sim::entities::NpcAction::Wait;
    }
}

void NpcFiniteStateMachine::TransitionTo(const sim::entities::NpcState nextState)
{
    state_ = nextState;
    timeInState_ = 0.0f;
}

} // namespace sim::ai
