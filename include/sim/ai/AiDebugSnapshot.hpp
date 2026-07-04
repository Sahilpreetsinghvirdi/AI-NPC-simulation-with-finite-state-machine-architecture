#pragma once

#include "sim/ai/ITrainablePolicy.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace sim::ai {

struct WeightStatistics {
    float minWeight{0.0f};
    float maxWeight{0.0f};
    float meanAbsWeight{0.0f};
    float l2Norm{0.0f};
    float parameterDrift{0.0f};
    float maxParameterUpdate{0.0f};
};

struct AiDebugSnapshot {
    std::uint64_t simulationTick{0};
    float elapsedSeconds{0.0f};
    float fps{0.0f};
    std::size_t activePolice{0};

    TrainingState trainingState;
    std::size_t rolloutSize{0};
    std::size_t rolloutCapacity{0};
    int lastAction{0};
    float lastReward{0.0f};
    float lastValue{0.0f};
    float lastNextValue{0.0f};
    float lastTdError{0.0f};
    float lastLogProbability{0.0f};
    float lastReturn{0.0f};
    float lastAdvantage{0.0f};
    float learningRate{0.0f};
    float clipRange{0.0f};
    float entropyCoefficient{0.0f};

    std::vector<float> observation;
    std::vector<float> normalizedObservation;
    std::vector<float> hiddenActivations;
    std::vector<float> actorLogits;
    std::vector<float> actionProbabilities;
    std::vector<float> networkParameters;
    WeightStatistics weightStats;
};

bool WriteAiDebugSnapshot(const AiDebugSnapshot& snapshot, const std::filesystem::path& path);
bool ReadAiDebugSnapshot(const std::filesystem::path& path, AiDebugSnapshot& snapshot);

} // namespace sim::ai
