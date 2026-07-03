#pragma once

#include "sim/ai/PolicyTypes.hpp"

#include <vector>

namespace sim::ai {

class IPolicy {
public:
    virtual ~IPolicy() = default;

    [[nodiscard]] virtual PolicyDescriptor GetDescriptor() const = 0;
    virtual void Reset() = 0;
    [[nodiscard]] virtual PolicyResult Evaluate(const PolicyRequest& request) = 0;
    [[nodiscard]] virtual std::vector<PolicyResult> EvaluateBatch(const std::vector<PolicyRequest>& requests);
};

} // namespace sim::ai
