#include "sim/ai/AiDebugSnapshot.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

namespace sim::ai {

namespace {

void WriteVector(std::ostream& output, const char* key, const std::vector<float>& values)
{
    output << key << '=';
    for (const float value : values) {
        output << value << ' ';
    }
    output << '\n';
}

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

std::vector<float> ParseVector(const std::string& value)
{
    std::istringstream stream(value);
    std::vector<float> result;
    float item = 0.0f;
    while (stream >> item) {
        result.push_back(item);
    }
    return result;
}

} // namespace

bool WriteAiDebugSnapshot(const AiDebugSnapshot& snapshot, const std::filesystem::path& path)
{
    std::error_code errorCode;
    std::filesystem::create_directories(path.parent_path(), errorCode);
    if (errorCode) {
        return false;
    }

    const std::filesystem::path tempPath = path.string() + ".tmp";
    {
        std::ofstream output(tempPath, std::ios::out | std::ios::trunc);
        if (!output.is_open()) {
            return false;
        }

        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        output << "AI_DEBUG_SNAPSHOT 1\n";
        output << "simulationTick=" << snapshot.simulationTick << '\n';
        output << "elapsedSeconds=" << snapshot.elapsedSeconds << '\n';
        output << "fps=" << snapshot.fps << '\n';
        output << "activePolice=" << snapshot.activePolice << '\n';
        output << "trainingStepCount=" << snapshot.trainingState.trainingStepCount << '\n';
        output << "episodeCount=" << snapshot.trainingState.episodeCount << '\n';
        output << "totalReward=" << snapshot.trainingState.totalReward << '\n';
        output << "policyUpdateCount=" << snapshot.trainingState.policyUpdateCount << '\n';
        output << "lastMeanReturn=" << snapshot.trainingState.lastMeanReturn << '\n';
        output << "lastMeanAdvantage=" << snapshot.trainingState.lastMeanAdvantage << '\n';
        output << "lastPolicyLoss=" << snapshot.trainingState.lastPolicyLoss << '\n';
        output << "lastValueLoss=" << snapshot.trainingState.lastValueLoss << '\n';
        output << "lastEntropy=" << snapshot.trainingState.lastEntropy << '\n';
        output << "learnedPolicyEnabled=" << (snapshot.trainingState.learnedPolicyEnabled ? 1 : 0) << '\n';
        output << "rolloutSize=" << snapshot.rolloutSize << '\n';
        output << "rolloutCapacity=" << snapshot.rolloutCapacity << '\n';
        output << "lastAction=" << snapshot.lastAction << '\n';
        output << "lastReward=" << snapshot.lastReward << '\n';
        output << "lastValue=" << snapshot.lastValue << '\n';
        output << "lastNextValue=" << snapshot.lastNextValue << '\n';
        output << "lastTdError=" << snapshot.lastTdError << '\n';
        output << "lastLogProbability=" << snapshot.lastLogProbability << '\n';
        output << "lastReturn=" << snapshot.lastReturn << '\n';
        output << "lastAdvantage=" << snapshot.lastAdvantage << '\n';
        output << "learningRate=" << snapshot.learningRate << '\n';
        output << "clipRange=" << snapshot.clipRange << '\n';
        output << "entropyCoefficient=" << snapshot.entropyCoefficient << '\n';
        output << "lastTotalLoss=" << snapshot.trainingState.lastTotalLoss << '\n';
        output << "lastClipFraction=" << snapshot.trainingState.lastClipFraction << '\n';
        output << "lastKlDivergence=" << snapshot.trainingState.lastKlDivergence << '\n';
        output << "lastExplainedVariance=" << snapshot.trainingState.lastExplainedVariance << '\n';
        output << "lastGradientNorm=" << snapshot.trainingState.lastGradientNorm << '\n';
        output << "lastParameterUpdateNorm=" << snapshot.trainingState.lastParameterUpdateNorm << '\n';
        output << "lastMaxParameterUpdate=" << snapshot.trainingState.lastMaxParameterUpdate << '\n';
        output << "lastPpoEpoch=" << snapshot.trainingState.lastPpoEpoch << '\n';
        output << "lastPpoMinibatch=" << snapshot.trainingState.lastPpoMinibatch << '\n';
        output << "minWeight=" << snapshot.weightStats.minWeight << '\n';
        output << "maxWeight=" << snapshot.weightStats.maxWeight << '\n';
        output << "meanAbsWeight=" << snapshot.weightStats.meanAbsWeight << '\n';
        output << "weightL2Norm=" << snapshot.weightStats.l2Norm << '\n';
        output << "parameterDrift=" << snapshot.weightStats.parameterDrift << '\n';
        output << "maxParameterUpdate=" << snapshot.weightStats.maxParameterUpdate << '\n';
        WriteVector(output, "observation", snapshot.observation);
        WriteVector(output, "normalizedObservation", snapshot.normalizedObservation);
        WriteVector(output, "hiddenActivations", snapshot.hiddenActivations);
        WriteVector(output, "actorLogits", snapshot.actorLogits);
        WriteVector(output, "actionProbabilities", snapshot.actionProbabilities);
        WriteVector(output, "networkParameters", snapshot.networkParameters);
    }

    std::filesystem::rename(tempPath, path, errorCode);
    if (errorCode) {
        std::filesystem::remove(path, errorCode);
        errorCode.clear();
        std::filesystem::rename(tempPath, path, errorCode);
    }

    return !errorCode;
}

bool ReadAiDebugSnapshot(const std::filesystem::path& path, AiDebugSnapshot& snapshot)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }

    std::string header;
    int version = 0;
    input >> header >> version;
    if (header != "AI_DEBUG_SNAPSHOT" || version != 1) {
        return false;
    }

    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        std::string key;
        std::string value;
        if (!ParseKeyValue(line, key, value)) {
            continue;
        }

        if (key == "simulationTick") snapshot.simulationTick = std::stoull(value);
        else if (key == "elapsedSeconds") snapshot.elapsedSeconds = std::stof(value);
        else if (key == "fps") snapshot.fps = std::stof(value);
        else if (key == "activePolice") snapshot.activePolice = static_cast<std::size_t>(std::stoull(value));
        else if (key == "trainingStepCount") snapshot.trainingState.trainingStepCount = std::stoull(value);
        else if (key == "episodeCount") snapshot.trainingState.episodeCount = std::stoull(value);
        else if (key == "totalReward") snapshot.trainingState.totalReward = std::stof(value);
        else if (key == "policyUpdateCount") snapshot.trainingState.policyUpdateCount = std::stoull(value);
        else if (key == "lastMeanReturn") snapshot.trainingState.lastMeanReturn = std::stof(value);
        else if (key == "lastMeanAdvantage") snapshot.trainingState.lastMeanAdvantage = std::stof(value);
        else if (key == "lastPolicyLoss") snapshot.trainingState.lastPolicyLoss = std::stof(value);
        else if (key == "lastValueLoss") snapshot.trainingState.lastValueLoss = std::stof(value);
        else if (key == "lastEntropy") snapshot.trainingState.lastEntropy = std::stof(value);
        else if (key == "learnedPolicyEnabled") snapshot.trainingState.learnedPolicyEnabled = value == "1";
        else if (key == "rolloutSize") snapshot.rolloutSize = static_cast<std::size_t>(std::stoull(value));
        else if (key == "rolloutCapacity") snapshot.rolloutCapacity = static_cast<std::size_t>(std::stoull(value));
        else if (key == "lastAction") snapshot.lastAction = std::stoi(value);
        else if (key == "lastReward") snapshot.lastReward = std::stof(value);
        else if (key == "lastValue") snapshot.lastValue = std::stof(value);
        else if (key == "lastNextValue") snapshot.lastNextValue = std::stof(value);
        else if (key == "lastTdError") snapshot.lastTdError = std::stof(value);
        else if (key == "lastLogProbability") snapshot.lastLogProbability = std::stof(value);
        else if (key == "lastReturn") snapshot.lastReturn = std::stof(value);
        else if (key == "lastAdvantage") snapshot.lastAdvantage = std::stof(value);
        else if (key == "learningRate") snapshot.learningRate = std::stof(value);
        else if (key == "clipRange") snapshot.clipRange = std::stof(value);
        else if (key == "entropyCoefficient") snapshot.entropyCoefficient = std::stof(value);
        else if (key == "lastTotalLoss") snapshot.trainingState.lastTotalLoss = std::stof(value);
        else if (key == "lastClipFraction") snapshot.trainingState.lastClipFraction = std::stof(value);
        else if (key == "lastKlDivergence") snapshot.trainingState.lastKlDivergence = std::stof(value);
        else if (key == "lastExplainedVariance") snapshot.trainingState.lastExplainedVariance = std::stof(value);
        else if (key == "lastGradientNorm") snapshot.trainingState.lastGradientNorm = std::stof(value);
        else if (key == "lastParameterUpdateNorm") snapshot.trainingState.lastParameterUpdateNorm = std::stof(value);
        else if (key == "lastMaxParameterUpdate") snapshot.trainingState.lastMaxParameterUpdate = std::stof(value);
        else if (key == "lastPpoEpoch") snapshot.trainingState.lastPpoEpoch = static_cast<std::size_t>(std::stoull(value));
        else if (key == "lastPpoMinibatch") snapshot.trainingState.lastPpoMinibatch = static_cast<std::size_t>(std::stoull(value));
        else if (key == "minWeight") snapshot.weightStats.minWeight = std::stof(value);
        else if (key == "maxWeight") snapshot.weightStats.maxWeight = std::stof(value);
        else if (key == "meanAbsWeight") snapshot.weightStats.meanAbsWeight = std::stof(value);
        else if (key == "weightL2Norm") snapshot.weightStats.l2Norm = std::stof(value);
        else if (key == "parameterDrift") snapshot.weightStats.parameterDrift = std::stof(value);
        else if (key == "maxParameterUpdate") snapshot.weightStats.maxParameterUpdate = std::stof(value);
        else if (key == "observation") snapshot.observation = ParseVector(value);
        else if (key == "normalizedObservation") snapshot.normalizedObservation = ParseVector(value);
        else if (key == "hiddenActivations") snapshot.hiddenActivations = ParseVector(value);
        else if (key == "actorLogits") snapshot.actorLogits = ParseVector(value);
        else if (key == "actionProbabilities") snapshot.actionProbabilities = ParseVector(value);
        else if (key == "networkParameters") snapshot.networkParameters = ParseVector(value);
    }

    return true;
}

} // namespace sim::ai
