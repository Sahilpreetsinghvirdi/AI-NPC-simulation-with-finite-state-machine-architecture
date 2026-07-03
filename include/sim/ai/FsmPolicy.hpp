#pragma once

#include "sim/ai/IPolicy.hpp"
#include "sim/ai/NpcFiniteStateMachine.hpp"

#include <mutex>

namespace sim::ai {

class FsmPolicy final : public IPolicy {
public:
    explicit FsmPolicy(sim::entities::NpcState initialState = sim::entities::NpcState::Idle);

    [[nodiscard]] PolicyDescriptor GetDescriptor() const override;
    void Reset() override;
    [[nodiscard]] PolicyResult Evaluate(const PolicyRequest& request) override;

private:
    mutable std::mutex mutex_;
    NpcFiniteStateMachine stateMachine_;
    sim::entities::NpcState initialState_{sim::entities::NpcState::Idle};
};

} // namespace sim::ai
