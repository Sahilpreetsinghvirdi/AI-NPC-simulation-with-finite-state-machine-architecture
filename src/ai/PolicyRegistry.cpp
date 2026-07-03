#include "sim/ai/PolicyRegistry.hpp"

#include "sim/ai/FsmPolicy.hpp"
#include "sim/ai/PersistentLearningPolicy.hpp"

#include <utility>

namespace sim::ai {

std::unique_ptr<IPolicy> PolicyRegistry::Create(const PolicyConfig& config)
{
    return Create(config.policyId);
}

std::unique_ptr<IPolicy> PolicyRegistry::Create(std::string policyId)
{
    {
        std::lock_guard lock(Mutex());
        const auto iter = Entries().find(policyId);
        if (iter != Entries().end()) {
            return iter->second.factory();
        }
    }

    return CreateBuiltIn(policyId);
}

bool PolicyRegistry::Register(const PolicyDescriptor& descriptor, Factory factory)
{
    if (descriptor.id.empty() || !factory) {
        return false;
    }

    std::lock_guard lock(Mutex());
    Entries()[descriptor.id] = Entry{
        .descriptor = descriptor,
        .factory = std::move(factory)
    };
    return true;
}

std::vector<PolicyDescriptor> PolicyRegistry::ListPolicies()
{
    std::vector<PolicyDescriptor> policies;
    policies.push_back(FsmPolicy{}.GetDescriptor());
    policies.push_back(PersistentLearningPolicy{}.GetDescriptor());

    std::lock_guard lock(Mutex());
    policies.reserve(policies.size() + Entries().size());
    for (const auto& [id, entry] : Entries()) {
        (void)id;
        policies.push_back(entry.descriptor);
    }

    return policies;
}

std::unique_ptr<IPolicy> PolicyRegistry::CreateBuiltIn(const std::string& policyId)
{
    if (policyId == "fsm") {
        return std::make_unique<FsmPolicy>();
    }

    if (policyId == "persistent_learning_fsm") {
        return std::make_unique<PersistentLearningPolicy>();
    }

    return std::make_unique<FsmPolicy>();
}

std::unordered_map<std::string, PolicyRegistry::Entry>& PolicyRegistry::Entries()
{
    static std::unordered_map<std::string, Entry> entries;
    return entries;
}

std::mutex& PolicyRegistry::Mutex()
{
    static std::mutex mutex;
    return mutex;
}

} // namespace sim::ai
