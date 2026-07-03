#include "sim/rl/CheckpointManager.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

namespace sim::rl {

namespace {

constexpr std::uint32_t kCheckpointFormatVersion = 2;
constexpr const char* kCheckpointHeader = "AI_NPC_POLICY_CHECKPOINT";

bool ParseKeyValue(const std::string& line, std::string& key, std::string& value)
{
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos) {
        return false;
    }

    key = line.substr(0, separator);
    value = line.substr(separator + 1);
    return true;
}

} // namespace

CheckpointManager::CheckpointManager(std::filesystem::path checkpointPath)
    : checkpointPath_(std::move(checkpointPath))
{
}

const std::filesystem::path& CheckpointManager::GetCheckpointPath() const
{
    return checkpointPath_;
}

bool CheckpointManager::Exists() const
{
    std::error_code errorCode;
    return std::filesystem::exists(checkpointPath_, errorCode);
}

CheckpointResult CheckpointManager::Load(sim::ai::ITrainablePolicy& policy) const
{
    std::ifstream input(checkpointPath_);
    if (!input.is_open()) {
        return {
            .ok = false,
            .message = "Checkpoint does not exist or cannot be opened: " + checkpointPath_.string()
        };
    }

    std::string header;
    std::uint32_t version = 0;
    input >> header >> version;
    if (header != kCheckpointHeader || version < 1 || version > kCheckpointFormatVersion) {
        return {
            .ok = false,
            .message = "Unsupported checkpoint format or version."
        };
    }

    sim::ai::PolicyCheckpoint checkpoint;
    checkpoint.formatVersion = version;

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::string key;
        std::string value;
        if (!ParseKeyValue(line, key, value)) {
            continue;
        }

        if (key == "policyId") {
            checkpoint.trainingState.policyId = value;
        } else if (key == "algorithmId") {
            checkpoint.trainingState.algorithmId = value;
        } else if (key == "trainingStepCount") {
            checkpoint.trainingState.trainingStepCount = std::stoull(value);
        } else if (key == "episodeCount") {
            checkpoint.trainingState.episodeCount = std::stoull(value);
        } else if (key == "totalReward") {
            checkpoint.trainingState.totalReward = std::stof(value);
        } else if (key == "policyUpdateCount") {
            checkpoint.trainingState.policyUpdateCount = std::stoull(value);
        } else if (key == "lastMeanReturn") {
            checkpoint.trainingState.lastMeanReturn = std::stof(value);
        } else if (key == "lastMeanAdvantage") {
            checkpoint.trainingState.lastMeanAdvantage = std::stof(value);
        } else if (key == "learnedPolicyEnabled") {
            checkpoint.trainingState.learnedPolicyEnabled = value == "1";
        } else if (key == "networkInputSize") {
            checkpoint.network.inputSize = static_cast<std::size_t>(std::stoull(value));
        } else if (key == "networkHiddenSize") {
            checkpoint.network.hiddenSize = static_cast<std::size_t>(std::stoull(value));
        } else if (key == "networkOutputSize") {
            checkpoint.network.outputSize = static_cast<std::size_t>(std::stoull(value));
        } else if (key == "networkParameters") {
            std::istringstream stream(value);
            float parameter = 0.0f;
            while (stream >> parameter) {
                checkpoint.network.parameters.push_back(parameter);
            }
        } else if (key.rfind("action.", 0) == 0) {
            std::istringstream stream(value);
            int action = 0;
            sim::ai::ActionValueEstimate estimate;
            stream >> action >> estimate.value >> estimate.sampleCount;
            checkpoint.actionValues[action] = estimate;
        }
    }

    if (!policy.ImportCheckpoint(checkpoint)) {
        return {
            .ok = false,
            .message = "Policy rejected checkpoint data."
        };
    }

    return {
        .ok = true,
        .message = "Checkpoint loaded: " + checkpointPath_.string()
    };
}

CheckpointResult CheckpointManager::Save(const sim::ai::ITrainablePolicy& policy) const
{
    std::error_code errorCode;
    std::filesystem::create_directories(checkpointPath_.parent_path(), errorCode);
    if (errorCode) {
        return {
            .ok = false,
            .message = "Failed to create checkpoint directory: " + errorCode.message()
        };
    }

    const sim::ai::PolicyCheckpoint checkpoint = policy.ExportCheckpoint();
    const std::filesystem::path tempPath = checkpointPath_.string() + ".tmp";

    {
        std::ofstream output(tempPath, std::ios::out | std::ios::trunc);
        if (!output.is_open()) {
            return {
                .ok = false,
                .message = "Failed to open checkpoint for writing: " + tempPath.string()
            };
        }

        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        output << kCheckpointHeader << ' ' << kCheckpointFormatVersion << '\n';
        output << "policyId=" << checkpoint.trainingState.policyId << '\n';
        output << "algorithmId=" << checkpoint.trainingState.algorithmId << '\n';
        output << "trainingStepCount=" << checkpoint.trainingState.trainingStepCount << '\n';
        output << "episodeCount=" << checkpoint.trainingState.episodeCount << '\n';
        output << "totalReward=" << checkpoint.trainingState.totalReward << '\n';
        output << "policyUpdateCount=" << checkpoint.trainingState.policyUpdateCount << '\n';
        output << "lastMeanReturn=" << checkpoint.trainingState.lastMeanReturn << '\n';
        output << "lastMeanAdvantage=" << checkpoint.trainingState.lastMeanAdvantage << '\n';
        output << "learnedPolicyEnabled=" << (checkpoint.trainingState.learnedPolicyEnabled ? 1 : 0) << '\n';
        output << "networkInputSize=" << checkpoint.network.inputSize << '\n';
        output << "networkHiddenSize=" << checkpoint.network.hiddenSize << '\n';
        output << "networkOutputSize=" << checkpoint.network.outputSize << '\n';
        output << "networkParameters=";
        for (const float parameter : checkpoint.network.parameters) {
            output << parameter << ' ';
        }
        output << '\n';

        std::size_t index = 0;
        for (const auto& [action, estimate] : checkpoint.actionValues) {
            output << "action." << index << '=' << action << ' ' << estimate.value << ' '
                   << estimate.sampleCount << '\n';
            ++index;
        }
    }

    std::filesystem::rename(tempPath, checkpointPath_, errorCode);
    if (errorCode) {
        std::filesystem::remove(checkpointPath_, errorCode);
        errorCode.clear();
        std::filesystem::rename(tempPath, checkpointPath_, errorCode);
    }

    if (errorCode) {
        return {
            .ok = false,
            .message = "Failed to finalize checkpoint: " + errorCode.message()
        };
    }

    return {
        .ok = true,
        .message = "Checkpoint saved: " + checkpointPath_.string()
    };
}

} // namespace sim::rl
