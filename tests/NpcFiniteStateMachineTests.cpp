#include "sim/ai/NpcFiniteStateMachine.hpp"
#include "sim/Simulation.hpp"
#include "sim/core/Logger.hpp"
#include "sim/entities/Player.hpp"
#include "sim/entities/PoliceManager.hpp"
#include "sim/entities/PoliceNpc.hpp"
#include "sim/math/Vec2.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

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

bool PoliceSpeedScalesWithWantedLevel()
{
    sim::entities::Player player({120.0f, 0.0f});
    sim::entities::PoliceNpc police({0.0f, 0.0f});

    player.SetWantedLevel(3);
    police.Update(0.1f, player);
    const bool wantedThreeOk = Expect(AlmostEqual(police.GetCurrentSpeed(), sim::entities::PoliceNpc::kBaseSpeed),
                                     "wanted 3 should keep base police speed");

    player.SetWantedLevel(4);
    police.Update(0.1f, player);
    const float wantedFourSpeed = sim::entities::PoliceNpc::kBaseSpeed *
                                  (1.0f + sim::entities::PoliceNpc::kSpeedIncreasePerStar);
    const bool wantedFourOk = Expect(AlmostEqual(police.GetCurrentSpeed(), wantedFourSpeed),
                                    "wanted 4 should increase police speed by one configured step");

    player.SetWantedLevel(sim::entities::Player::kMaxWantedLevel);
    police.Update(0.1f, player);
    const bool wantedFiveOk = Expect(sim::entities::PoliceNpc::kMaxPoliceSpeed > 18.0f,
                                    "configured max police speed should exceed player max speed") &&
                              Expect(police.GetCurrentSpeed() > wantedFourSpeed,
                                     "max wanted should ramp police speed above wanted 4 speed") &&
                              Expect(police.GetCurrentSpeed() <= sim::entities::PoliceNpc::kMaxPoliceSpeed,
                                     "current police speed should not exceed configured max speed");

    return wantedThreeOk && wantedFourOk && wantedFiveOk;
}

bool PoliceManagerSpawnsAndUpdatesIndependentPolice()
{
    sim::entities::Player player({0.0f, 0.0f});
    player.SetWantedLevel(3);

    sim::entities::PoliceManager manager;
    manager.SyncToWantedLevel(player.GetWantedLevel(), player.GetPosition());

    const bool countOk = Expect(manager.GetActiveCount() == 3, "wanted 3 should spawn three police NPCs");
    const auto& policeNpcs = manager.GetPoliceNpcs();
    const bool uniqueSpawnOk = Expect(policeNpcs.size() == 3 &&
                                          !sim::math::ApproxEqual(policeNpcs[0].GetPosition(), policeNpcs[1].GetPosition()) &&
                                          !sim::math::ApproxEqual(policeNpcs[1].GetPosition(), policeNpcs[2].GetPosition()),
                                      "spawned police should not share one position");

    manager.UpdateAll(0.1f, player);

    bool allUpdated = true;
    for (const auto& police : manager.GetPoliceNpcs()) {
        allUpdated = allUpdated &&
                     police.GetState() == sim::entities::NpcState::Pursue &&
                     police.GetLastAction() == sim::entities::NpcAction::MoveTowardPlayer;
    }

    const bool updatedOk = Expect(allUpdated, "each police NPC should update its own FSM and action");

    manager.SyncToWantedLevel(1, player.GetPosition());
    const bool shrinkOk = Expect(manager.GetActiveCount() == 1, "wanted 1 should shrink active police to one");

    return countOk && uniqueSpawnOk && updatedOk && shrinkOk;
}

bool PoliceInterceptionChangesRelativePositions()
{
    sim::entities::Player player({20.0f, 20.0f});
    player.SetWantedLevel(5);

    sim::entities::PoliceManager manager;
    manager.SyncToWantedLevel(player.GetWantedLevel(), player.GetPosition());

    const sim::math::Vec2 playerVelocity{8.0f, 3.0f};
    manager.UpdateAll(0.1f, player, playerVelocity);

    const auto& firstFramePolice = manager.GetPoliceNpcs();
    const bool countOk = Expect(firstFramePolice.size() == 5, "wanted 5 should create five police roles");
    const bool targetSpreadOk = Expect(!sim::math::ApproxEqual(firstFramePolice[0].GetCurrentTarget(),
                                                               firstFramePolice[1].GetCurrentTarget()) &&
                                           !sim::math::ApproxEqual(firstFramePolice[1].GetCurrentTarget(),
                                                                   firstFramePolice[2].GetCurrentTarget()),
                                       "interception roles should produce different targets");

    std::vector<sim::math::Vec2> initialOffsets;
    for (const auto& police : firstFramePolice) {
        initialOffsets.push_back(police.GetPosition() - player.GetPosition());
    }

    for (int step = 0; step < 20; ++step) {
        player.Translate(playerVelocity * 0.1f);
        manager.UpdateAll(0.1f, player, playerVelocity);
    }

    bool relativePositionsChanged = false;
    const auto& finalPolice = manager.GetPoliceNpcs();
    for (std::size_t index = 0; index < finalPolice.size() && index < initialOffsets.size(); ++index) {
        const sim::math::Vec2 currentOffset = finalPolice[index].GetPosition() - player.GetPosition();
        if (!sim::math::ApproxEqual(currentOffset, initialOffsets[index], 0.5f)) {
            relativePositionsChanged = true;
            break;
        }
    }

    const bool relativeOk = Expect(relativePositionsChanged,
                                   "police/player relative positions should change during pursuit");

    return countOk && targetSpreadOk && relativeOk;
}

bool PoliceSteeringDoesNotInstantlySnap()
{
    sim::entities::Player player({-100.0f, 0.0f});
    player.SetWantedLevel(1);

    sim::entities::PoliceNpc police({0.0f, 0.0f});
    police.Update(0.1f, player, {-100.0f, 0.0f}, {}, true);

    const bool desiredOk = Expect(police.GetDesiredHeading().x < -0.9f,
                                  "desired heading should point toward the intercept target");
    const bool currentOk = Expect(police.GetCurrentHeading().x > 0.8f,
                                  "current heading should turn gradually instead of snapping backward");
    const bool steeringOk = Expect(police.GetSteeringVector().LengthSquared() > 0.0f,
                                   "steering vector should expose heading correction");

    return desiredOk && currentOk && steeringOk;
}

bool SimulationStartsWithOneWantedStarAndOnePolice()
{
    sim::core::Logger::Config logConfig;
    logConfig.logToConsole = false;
    logConfig.logToFile = false;
    sim::core::Logger logger(logConfig);

    sim::Simulation simulation(logger, {
        .maxTicks = 0,
        .tickRateHz = 10.0f,
        .realTimePacing = false
    });

    const bool initialWantedOk = Expect(simulation.GetPlayer().GetWantedLevel() == 1,
                                        "simulation should start at wanted level 1");
    const bool initialPoliceOk = Expect(simulation.GetPoliceManager().GetActiveCount() == 1,
                                        "simulation should start with exactly one police NPC");

    for (int tick = 0; tick < 16; ++tick) {
        simulation.Tick();
    }

    const bool noForcedWantedOk = Expect(simulation.GetPlayer().GetWantedLevel() == 1,
                                         "early simulation ticks should not force wanted level to 3");
    const bool noForcedPoliceOk = Expect(simulation.GetPoliceManager().GetActiveCount() == 1,
                                         "early simulation ticks should not spawn extra police");

    return initialWantedOk && initialPoliceOk && noForcedWantedOk && noForcedPoliceOk;
}

} // namespace

int main()
{
    const bool ok = PatrolsWhenPlayerIsNotWanted() &&
                    InvestigatesDistantWantedPlayer() &&
                    PursuesNearbyWantedPlayer() &&
                    AccumulatesTimeWhileStateIsStable() &&
                    PoliceNpcUsesStateMachineDecision() &&
                    PoliceSpeedScalesWithWantedLevel() &&
                    PoliceManagerSpawnsAndUpdatesIndependentPolice() &&
                    PoliceInterceptionChangesRelativePositions() &&
                    PoliceSteeringDoesNotInstantlySnap() &&
                    SimulationStartsWithOneWantedStarAndOnePolice();

    if (ok) {
        std::cout << "All NPC FSM tests passed.\n";
        return 0;
    }

    return 1;
}
