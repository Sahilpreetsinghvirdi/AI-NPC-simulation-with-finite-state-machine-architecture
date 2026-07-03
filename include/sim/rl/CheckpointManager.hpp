#pragma once

#include "sim/ai/ITrainablePolicy.hpp"

#include <filesystem>
#include <string>

namespace sim::rl {

struct CheckpointResult {
    bool ok{false};
    std::string message;
};

class CheckpointManager {
public:
    explicit CheckpointManager(std::filesystem::path checkpointPath);

    [[nodiscard]] const std::filesystem::path& GetCheckpointPath() const;
    [[nodiscard]] bool Exists() const;
    [[nodiscard]] CheckpointResult Load(sim::ai::ITrainablePolicy& policy) const;
    [[nodiscard]] CheckpointResult Save(const sim::ai::ITrainablePolicy& policy) const;

private:
    std::filesystem::path checkpointPath_;
};

} // namespace sim::rl
