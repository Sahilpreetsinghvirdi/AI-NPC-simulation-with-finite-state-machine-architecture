#pragma once

#include "sim/ai/NpcFiniteStateMachine.hpp"
#include "sim/ai/Observation.hpp"
#include "sim/entities/NpcAction.hpp"
#include "sim/entities/NpcState.hpp"
#include "sim/rl/RlTypes.hpp"

#include <string>

namespace sim::ai {

enum class PolicyStatus {
    Ok = 0,
    NotReady,
    InvalidRequest,
    Failed
};

struct PolicyConfig {
    std::string policyId{"fsm"};
    bool deterministic{true};
};

struct PolicyDescriptor {
    std::string id;
    std::string displayName;
    bool supportsBatching{false};
    bool supportsAsync{false};
    bool deterministicByDefault{true};
};

struct PolicyContext {
    sim::rl::RunMode mode{sim::rl::RunMode::Inference};
    sim::rl::StepMetadata metadata;
    bool deterministic{true};
    bool allowAsync{false};
    bool partOfBatch{false};
};

struct PolicyRequest {
    sim::rl::AgentHandle agent;
    sim::ai::Observation observation;
    sim::ai::NpcAiContext npcContext;
    float deltaTime{0.0f};
    PolicyContext context;
};

struct PolicyDecision {
    sim::entities::NpcState state{sim::entities::NpcState::Idle};
    sim::entities::NpcAction action{sim::entities::NpcAction::None};
    sim::rl::Action rawAction{sim::rl::Action::FromDiscrete(static_cast<int>(sim::entities::NpcAction::None))};
    float confidence{1.0f};
    bool stateChanged{false};
    float timeInState{0.0f};
};

struct PolicyResult {
    PolicyStatus status{PolicyStatus::Ok};
    PolicyDecision decision;
    std::string message;

    [[nodiscard]] bool IsOk() const { return status == PolicyStatus::Ok; }
};

} // namespace sim::ai
