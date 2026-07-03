#pragma once

#include "sim/ai/IPolicy.hpp"
#include "sim/ai/PolicyTypes.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace sim::ai {

class PolicyRegistry {
public:
    using Factory = std::function<std::unique_ptr<IPolicy>()>;

    static std::unique_ptr<IPolicy> Create(const PolicyConfig& config);
    static std::unique_ptr<IPolicy> Create(std::string policyId);
    static bool Register(const PolicyDescriptor& descriptor, Factory factory);
    static std::vector<PolicyDescriptor> ListPolicies();

private:
    struct Entry {
        PolicyDescriptor descriptor;
        Factory factory;
    };

    static std::unique_ptr<IPolicy> CreateBuiltIn(const std::string& policyId);
    static std::unordered_map<std::string, Entry>& Entries();
    static std::mutex& Mutex();
};

} // namespace sim::ai
