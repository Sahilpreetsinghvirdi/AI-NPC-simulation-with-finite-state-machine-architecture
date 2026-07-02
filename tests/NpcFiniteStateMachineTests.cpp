#include "sim/ai/NpcFiniteStateMachine.hpp"
#include "sim/entities/Player.hpp"
#include "sim/entities/PoliceNpc.hpp"
#include "sim/math/Vec2.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

bool Expect(const bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

bool AlmostEqual(const float a, const float b, const float epsilon = 1e-4f)
{
    return std::fabs(a - b) <= epsilon;
}

sim::ai::NpcAiContext MakeContext(const sim::math::Vec2 npcPosition,
                                  const sim::math::Vec2 playerPosition,
                                  const int wantedLevel,
                                  const bool playerAlive = true)
{
    return {
        .npcPosition = npcPosition,
        .playerPosition = playerPosition,
        .playerWantedLevel = wantedLevel,
        .playerAlive = playerAlive
    };
}

bool PatrolsWhenPlayerIsNotWanted()
{
    sim::ai::NpcFiniteStateMachine fsm;

    const auto decision = fsm.Update(MakeContext({0.0f, 0.0f}, {20.0f, 0.0f}, 0), 0.1f);

    return Expect(decision.state == sim::entities::NpcState::Patrol, "unwanted player should put NPC on patrol") &&
           Expect(decision.action == sim::entities::NpcAction::PatrolStep, "patrol state should patrol step") &&
           Expect(decision.stateChanged, "initial idle to patrol transition should be reported") &&
           Expect(AlmostEqual(decision.timeInState, 0.0f), "new state should start at zero elapsed time");
}

bool InvestigatesDistantWantedPlayer()
{
    sim::ai::NpcFiniteStateMachine fsm;

    const auto decision = fsm.Update(MakeContext({0.0f, 0.0f}, {120.0f, 0.0f}, 2), 0.1f);

    return Expect(decision.state == sim::entities::NpcState::Investigate, "distant wanted player should be investigated") &&
           Expect(decision.action == sim::entities::NpcAction::MoveTowardPlayer,
                  "investigate action should move toward known player position");
}

bool PursuesNearbyWantedPlayer()
{
    sim::ai::NpcFiniteStateMachine fsm(sim::entities::NpcState::Patrol);

    const auto decision = fsm.Update(MakeContext({0.0f, 0.0f}, {50.0f, 0.0f}, 1), 0.1f);

    return Expect(decision.state == sim::entities::NpcState::Pursue, "nearby wanted player should be pursued") &&
           Expect(decision.action == sim::entities::NpcAction::MoveTowardPlayer, "pursue action should move toward player");
}

bool AccumulatesTimeWhileStateIsStable()
{
    sim::ai::NpcFiniteStateMachine fsm;
    const auto context = MakeContext({0.0f, 0.0f}, {20.0f, 0.0f}, 0);

    fsm.Update(context, 0.1f);
    const auto decision = fsm.Update(context, 0.25f);

    return Expect(!decision.stateChanged, "stable state should not report a transition") &&
           Expect(AlmostEqual(decision.timeInState, 0.25f), "stable state should accumulate elapsed time");
}

bool PoliceNpcUsesStateMachineDecision()
{
    sim::entities::Player player({10.0f, 0.0f});
    player.SetWantedLevel(1);

    sim::entities::PoliceNpc police({0.0f, 0.0f});
    police.Update(0.1f, player);

    return Expect(police.GetState() == sim::entities::NpcState::Pursue, "police should expose FSM pursue state") &&
           Expect(police.GetLastAction() == sim::entities::NpcAction::MoveTowardPlayer ||
                      police.GetLastAction() == sim::entities::NpcAction::Attack,
                  "police should act from FSM or attack when in range");
}

} // namespace

int main()
{
    const bool ok = PatrolsWhenPlayerIsNotWanted() &&
                    InvestigatesDistantWantedPlayer() &&
                    PursuesNearbyWantedPlayer() &&
                    AccumulatesTimeWhileStateIsStable() &&
                    PoliceNpcUsesStateMachineDecision();

    if (ok) {
        std::cout << "All NPC FSM tests passed.\n";
        return 0;
    }

    return 1;
}
