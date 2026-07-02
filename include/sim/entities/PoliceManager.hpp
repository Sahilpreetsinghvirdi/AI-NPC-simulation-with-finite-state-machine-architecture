#pragma once

#include "sim/entities/NpcState.hpp"
#include "sim/entities/Player.hpp"
#include "sim/entities/PoliceNpc.hpp"
#include "sim/math/Vec2.hpp"

#include <cstddef>
#include <vector>

namespace sim::entities {

class PoliceManager {
public:
    void SyncToWantedLevel(int wantedLevel, sim::math::Vec2 playerPosition);
    void UpdateAll(float deltaTime, Player& player, sim::math::Vec2 playerVelocity = {});

    [[nodiscard]] std::size_t GetActiveCount() const;
    [[nodiscard]] bool HasActivePolice() const;
    [[nodiscard]] bool AnyInState(NpcState state) const;
    [[nodiscard]] const std::vector<PoliceNpc>& GetPoliceNpcs() const;
    [[nodiscard]] const PoliceNpc* GetPrimaryPolice() const;
    [[nodiscard]] const PoliceNpc* GetNearestPolice(sim::math::Vec2 position) const;
    [[nodiscard]] PoliceNpc* GetNearestPolice(sim::math::Vec2 position);

private:
    [[nodiscard]] sim::math::Vec2 CalculateSpawnPosition(sim::math::Vec2 playerPosition, std::size_t index) const;
    [[nodiscard]] sim::math::Vec2 CalculateInterceptTarget(const PoliceNpc& police,
                                                           const Player& player,
                                                           sim::math::Vec2 playerVelocity,
                                                           std::size_t index) const;
    [[nodiscard]] sim::math::Vec2 CalculateSeparationForce(std::size_t index) const;

    std::vector<PoliceNpc> policeNpcs_;
    float playerDamageCooldown_{0.0f};
};

} // namespace sim::entities
