#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sim::ai {

struct NetworkEvaluation {
    std::vector<float> logits;
    std::vector<float> probabilities;
    float value{0.0f};
    float entropy{0.0f};
};

class FeedForwardNetwork {
public:
    struct Config {
        std::size_t inputSize{8};
        std::size_t hiddenSize{16};
        std::size_t outputSize{5};
        std::uint32_t seed{1337};
    };

    FeedForwardNetwork();
    explicit FeedForwardNetwork(Config config);

    [[nodiscard]] const Config& GetConfig() const;
    [[nodiscard]] NetworkEvaluation Evaluate(const std::vector<float>& input) const;
    [[nodiscard]] std::vector<float> ForwardLogits(const std::vector<float>& input) const;
    [[nodiscard]] std::vector<float> ForwardProbabilities(const std::vector<float>& input) const;
    [[nodiscard]] float ForwardValue(const std::vector<float>& input) const;
    [[nodiscard]] int SelectGreedyAction(const std::vector<float>& input) const;
    [[nodiscard]] float LogProbability(const std::vector<float>& input, int action) const;

    void ApplyPolicyGradient(const std::vector<float>& input, int action, float advantage, float learningRate);
    void ApplyActorCriticGradient(const std::vector<float>& input,
                                  int action,
                                  float policyGradientScale,
                                  float valueTarget,
                                  float valueLossCoefficient,
                                  float entropyCoefficient,
                                  float learningRate);

    [[nodiscard]] std::vector<float> ExportParameters() const;
    bool ImportParameters(const std::vector<float>& parameters);
    [[nodiscard]] std::size_t ParameterCount() const;

private:
    [[nodiscard]] std::vector<float> NormalizeInput(const std::vector<float>& input) const;
    void InitializeParameters(std::uint32_t seed);

    Config config_;
    std::vector<float> inputHiddenWeights_;
    std::vector<float> hiddenBias_;
    std::vector<float> hiddenOutputWeights_;
    std::vector<float> outputBias_;
    std::vector<float> hiddenValueWeights_;
    float valueBias_{0.0f};
};

} // namespace sim::ai
