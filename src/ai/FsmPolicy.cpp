#include "sim/ai/FsmPolicy.hpp"

namespace sim::ai {

FsmPolicy::FsmPolicy(const sim::entities::NpcState initialState)
    : stateMachine_(initialState)
    , initialState_(initialState)
{
}

PolicyDescriptor FsmPolicy::GetDescriptor() const
{
    return {
        .id = "fsm",
        .displayName = "Finite State Machine Baseline",
        .supportsBatching = false,
        .supportsAsync = false,
        .deterministicByDefault = true
    };
}

void FsmPolicy::Reset()
{
    std::lock_guard lock(mutex_);
    stateMachine_.Reset(initialState_);
}

PolicyResult FsmPolicy::Evaluate(const PolicyRequest& request)
{
    if (request.deltaTime < 0.0f) {
        return {
            .status = PolicyStatus::InvalidRequest,
            .decision = {},
            .message = "Policy request deltaTime cannot be negative."
        };
    }

    std::lock_guard lock(mutex_);
    const NpcDecision fsmDecision = stateMachine_.Update(request.npcContext, request.deltaTime);

    return {
        .status = PolicyStatus::Ok,
        .decision = {
            .state = fsmDecision.state,
            .action = fsmDecision.action,
            .rawAction = sim::rl::Action::FromDiscrete(static_cast<int>(fsmDecision.action)),
            .confidence = 1.0f,
            .stateChanged = fsmDecision.stateChanged,
            .timeInState = fsmDecision.timeInState
        },
        .message = {}
    };
}

} // namespace sim::ai
