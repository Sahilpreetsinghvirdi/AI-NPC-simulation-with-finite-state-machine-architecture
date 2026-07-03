#include "sim/ai/IPolicy.hpp"

namespace sim::ai {

std::vector<PolicyResult> IPolicy::EvaluateBatch(const std::vector<PolicyRequest>& requests)
{
    std::vector<PolicyResult> results;
    results.reserve(requests.size());

    for (const PolicyRequest& request : requests) {
        results.push_back(Evaluate(request));
    }

    return results;
}

} // namespace sim::ai
