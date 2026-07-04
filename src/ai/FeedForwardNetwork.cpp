#include "sim/ai/FeedForwardNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sim::ai {

namespace {

float NextRandom(std::uint32_t& state)
{
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return (static_cast<float>(state & 0x00FFFFFFU) / static_cast<float>(0x00FFFFFFU)) - 0.5f;
}

std::vector<float> Softmax(const std::vector<float>& logits)
{
    if (logits.empty()) {
        return {};
    }

    const float maxLogit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size(), 0.0f);
    float sum = 0.0f;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - maxLogit);
        sum += probabilities[index];
    }

    if (sum <= std::numeric_limits<float>::epsilon()) {
        const float uniform = 1.0f / static_cast<float>(logits.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
        return probabilities;
    }

    for (float& probability : probabilities) {
        probability /= sum;
    }
    return probabilities;
}

} // namespace

FeedForwardNetwork::FeedForwardNetwork()
    : FeedForwardNetwork(Config{})
{
}

FeedForwardNetwork::FeedForwardNetwork(Config config)
    : config_(config)
{
    inputHiddenWeights_.resize(config_.inputSize * config_.hiddenSize);
    hiddenBias_.resize(config_.hiddenSize);
    hiddenOutputWeights_.resize(config_.hiddenSize * config_.outputSize);
    outputBias_.resize(config_.outputSize);
    hiddenValueWeights_.resize(config_.hiddenSize);
    InitializeParameters(config_.seed);
}

const FeedForwardNetwork::Config& FeedForwardNetwork::GetConfig() const
{
    return config_;
}

NetworkEvaluation FeedForwardNetwork::Evaluate(const std::vector<float>& input) const
{
    const std::vector<float> normalizedInput = NormalizeInput(input);
    std::vector<float> hidden(config_.hiddenSize, 0.0f);
    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        float value = hiddenBias_[hiddenIndex];
        for (std::size_t inputIndex = 0; inputIndex < config_.inputSize; ++inputIndex) {
            value += normalizedInput[inputIndex] * inputHiddenWeights_[hiddenIndex * config_.inputSize + inputIndex];
        }
        hidden[hiddenIndex] = std::tanh(value);
    }

    NetworkEvaluation evaluation;
    evaluation.normalizedInput = normalizedInput;
    evaluation.hiddenActivations = hidden;
    evaluation.logits.resize(config_.outputSize, 0.0f);
    for (std::size_t outputIndex = 0; outputIndex < config_.outputSize; ++outputIndex) {
        float value = outputBias_[outputIndex];
        for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
            value += hidden[hiddenIndex] * hiddenOutputWeights_[outputIndex * config_.hiddenSize + hiddenIndex];
        }
        evaluation.logits[outputIndex] = value;
    }

    evaluation.probabilities = Softmax(evaluation.logits);
    for (const float probability : evaluation.probabilities) {
        if (probability > 1.0e-6f) {
            evaluation.entropy -= probability * std::log(probability);
        }
    }

    evaluation.value = valueBias_;
    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        evaluation.value += hidden[hiddenIndex] * hiddenValueWeights_[hiddenIndex];
    }

    return evaluation;
}

std::vector<float> FeedForwardNetwork::ForwardLogits(const std::vector<float>& input) const
{
    return Evaluate(input).logits;
}

std::vector<float> FeedForwardNetwork::ForwardProbabilities(const std::vector<float>& input) const
{
    return Evaluate(input).probabilities;
}

float FeedForwardNetwork::ForwardValue(const std::vector<float>& input) const
{
    return Evaluate(input).value;
}

int FeedForwardNetwork::SelectGreedyAction(const std::vector<float>& input) const
{
    const std::vector<float> probabilities = ForwardProbabilities(input);
    if (probabilities.empty()) {
        return 0;
    }

    return static_cast<int>(std::distance(probabilities.begin(),
                                          std::max_element(probabilities.begin(), probabilities.end())));
}

float FeedForwardNetwork::LogProbability(const std::vector<float>& input, const int action) const
{
    const std::vector<float> probabilities = ForwardProbabilities(input);
    if (action < 0 || static_cast<std::size_t>(action) >= probabilities.size()) {
        return std::log(1.0e-6f);
    }

    return std::log(std::max(probabilities[static_cast<std::size_t>(action)], 1.0e-6f));
}

GradientUpdateStats FeedForwardNetwork::ApplyPolicyGradient(const std::vector<float>& input,
                                                            const int action,
                                                            const float advantage,
                                                            const float learningRate)
{
    return ApplyActorCriticGradient(input, action, advantage, 0.0f, 0.0f, 0.0f, learningRate);
}

GradientUpdateStats FeedForwardNetwork::ApplyActorCriticGradient(const std::vector<float>& input,
                                                                 const int action,
                                                                 const float policyGradientScale,
                                                                 const float valueTarget,
                                                                 const float valueLossCoefficient,
                                                                 const float entropyCoefficient,
                                                                 const float learningRate)
{
    if (action < 0 || static_cast<std::size_t>(action) >= config_.outputSize) {
        return {};
    }

    GradientUpdateStats stats;
    auto accumulateUpdate = [&](const float gradient) {
        const float update = learningRate * gradient;
        stats.gradientL2Norm += gradient * gradient;
        stats.parameterUpdateL2Norm += update * update;
        stats.maxParameterUpdate = std::max(stats.maxParameterUpdate, std::fabs(update));
        return update;
    };

    const std::vector<float> normalizedInput = NormalizeInput(input);
    std::vector<float> hiddenPreActivation(config_.hiddenSize, 0.0f);
    std::vector<float> hidden(config_.hiddenSize, 0.0f);
    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        float value = hiddenBias_[hiddenIndex];
        for (std::size_t inputIndex = 0; inputIndex < config_.inputSize; ++inputIndex) {
            value += normalizedInput[inputIndex] * inputHiddenWeights_[hiddenIndex * config_.inputSize + inputIndex];
        }
        hiddenPreActivation[hiddenIndex] = value;
        hidden[hiddenIndex] = std::tanh(value);
    }

    std::vector<float> logits(config_.outputSize, 0.0f);
    for (std::size_t outputIndex = 0; outputIndex < config_.outputSize; ++outputIndex) {
        float value = outputBias_[outputIndex];
        for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
            value += hidden[hiddenIndex] * hiddenOutputWeights_[outputIndex * config_.hiddenSize + hiddenIndex];
        }
        logits[outputIndex] = value;
    }

    float value = valueBias_;
    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        value += hidden[hiddenIndex] * hiddenValueWeights_[hiddenIndex];
    }

    std::vector<float> probabilities = Softmax(logits);
    std::vector<float> outputGradient(config_.outputSize, 0.0f);
    for (std::size_t outputIndex = 0; outputIndex < config_.outputSize; ++outputIndex) {
        const float target = outputIndex == static_cast<std::size_t>(action) ? 1.0f : 0.0f;
        const float entropyGradient = entropyCoefficient * (-(std::log(std::max(probabilities[outputIndex], 1.0e-6f)) + 1.0f));
        outputGradient[outputIndex] = ((target - probabilities[outputIndex]) * policyGradientScale) + entropyGradient;
    }

    std::vector<float> hiddenGradient(config_.hiddenSize, 0.0f);
    for (std::size_t outputIndex = 0; outputIndex < config_.outputSize; ++outputIndex) {
        for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
            hiddenGradient[hiddenIndex] +=
                outputGradient[outputIndex] * hiddenOutputWeights_[outputIndex * config_.hiddenSize + hiddenIndex];
            hiddenOutputWeights_[outputIndex * config_.hiddenSize + hiddenIndex] +=
                accumulateUpdate(outputGradient[outputIndex] * hidden[hiddenIndex]);
        }
        outputBias_[outputIndex] += accumulateUpdate(outputGradient[outputIndex]);
    }

    const float valueError = valueTarget - value;
    const float valueGradient = valueLossCoefficient * valueError;
    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        hiddenGradient[hiddenIndex] += valueGradient * hiddenValueWeights_[hiddenIndex];
        hiddenValueWeights_[hiddenIndex] += accumulateUpdate(valueGradient * hidden[hiddenIndex]);
    }
    valueBias_ += accumulateUpdate(valueGradient);

    for (std::size_t hiddenIndex = 0; hiddenIndex < config_.hiddenSize; ++hiddenIndex) {
        const float tanhDerivative = 1.0f - (hidden[hiddenIndex] * hidden[hiddenIndex]);
        const float gradient = hiddenGradient[hiddenIndex] * tanhDerivative;
        for (std::size_t inputIndex = 0; inputIndex < config_.inputSize; ++inputIndex) {
            inputHiddenWeights_[hiddenIndex * config_.inputSize + inputIndex] +=
                accumulateUpdate(gradient * normalizedInput[inputIndex]);
        }
        hiddenBias_[hiddenIndex] += accumulateUpdate(gradient);
    }

    stats.gradientL2Norm = std::sqrt(stats.gradientL2Norm);
    stats.parameterUpdateL2Norm = std::sqrt(stats.parameterUpdateL2Norm);
    return stats;
}

std::vector<float> FeedForwardNetwork::ExportParameters() const
{
    std::vector<float> parameters;
    parameters.reserve(ParameterCount());
    parameters.insert(parameters.end(), inputHiddenWeights_.begin(), inputHiddenWeights_.end());
    parameters.insert(parameters.end(), hiddenBias_.begin(), hiddenBias_.end());
    parameters.insert(parameters.end(), hiddenOutputWeights_.begin(), hiddenOutputWeights_.end());
    parameters.insert(parameters.end(), outputBias_.begin(), outputBias_.end());
    parameters.insert(parameters.end(), hiddenValueWeights_.begin(), hiddenValueWeights_.end());
    parameters.push_back(valueBias_);
    return parameters;
}

bool FeedForwardNetwork::ImportParameters(const std::vector<float>& parameters)
{
    if (parameters.size() != ParameterCount()) {
        return false;
    }

    std::size_t offset = 0;
    auto copyInto = [&](std::vector<float>& target) {
        std::copy(parameters.begin() + static_cast<std::ptrdiff_t>(offset),
                  parameters.begin() + static_cast<std::ptrdiff_t>(offset + target.size()),
                  target.begin());
        offset += target.size();
    };

    copyInto(inputHiddenWeights_);
    copyInto(hiddenBias_);
    copyInto(hiddenOutputWeights_);
    copyInto(outputBias_);
    copyInto(hiddenValueWeights_);
    valueBias_ = parameters[offset];
    return true;
}

std::size_t FeedForwardNetwork::ParameterCount() const
{
    return inputHiddenWeights_.size() + hiddenBias_.size() + hiddenOutputWeights_.size() + outputBias_.size() +
           hiddenValueWeights_.size() + 1;
}

std::vector<float> FeedForwardNetwork::NormalizeInput(const std::vector<float>& input) const
{
    std::vector<float> normalized(config_.inputSize, 0.0f);
    for (std::size_t index = 0; index < config_.inputSize && index < input.size(); ++index) {
        normalized[index] = std::clamp(input[index] / 120.0f, -4.0f, 4.0f);
    }
    return normalized;
}

void FeedForwardNetwork::InitializeParameters(std::uint32_t seed)
{
    const float inputScale = 1.0f / std::sqrt(static_cast<float>(std::max<std::size_t>(config_.inputSize, 1)));
    const float hiddenScale = 1.0f / std::sqrt(static_cast<float>(std::max<std::size_t>(config_.hiddenSize, 1)));
    for (float& weight : inputHiddenWeights_) {
        weight = NextRandom(seed) * inputScale;
    }
    std::fill(hiddenBias_.begin(), hiddenBias_.end(), 0.0f);
    for (float& weight : hiddenOutputWeights_) {
        weight = NextRandom(seed) * hiddenScale;
    }
    std::fill(outputBias_.begin(), outputBias_.end(), 0.0f);
    for (float& weight : hiddenValueWeights_) {
        weight = NextRandom(seed) * hiddenScale;
    }
    valueBias_ = 0.0f;
}

} // namespace sim::ai
