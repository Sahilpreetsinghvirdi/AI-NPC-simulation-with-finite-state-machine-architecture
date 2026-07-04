#include "sim/visualization/AiDebugWindows.hpp"

#include "sim/ai/AiDebugSnapshot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <raylib.h>
#include <string>
#include <vector>

namespace sim::visualization {

namespace {

constexpr Color kBg{18, 20, 26, 255};
constexpr Color kPanel{30, 33, 42, 255};
constexpr Color kText{228, 232, 240, 255};
constexpr Color kMuted{145, 152, 168, 255};
constexpr Color kGood{80, 210, 140, 230};
constexpr Color kBad{245, 100, 110, 230};
constexpr Color kAccent{105, 180, 255, 235};
constexpr Color kWarn{255, 190, 80, 235};

float Clamp01(const float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

void DrawMetric(const char* label, const char* value, const int x, int& y, const Color color = kText)
{
    DrawText(label, x, y, 16, kMuted);
    DrawText(value, x + 190, y, 16, color);
    y += 24;
}

void DrawGraph(const std::vector<float>& values, const Rectangle bounds, const Color color)
{
    DrawRectangleRec(bounds, {24, 26, 34, 255});
    DrawRectangleLinesEx(bounds, 1.0f, {55, 60, 72, 255});
    if (values.size() < 2) {
        return;
    }

    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const float minValue = *minIt;
    const float maxValue = *maxIt;
    const float range = std::max(maxValue - minValue, 1.0e-4f);
    for (std::size_t index = 1; index < values.size(); ++index) {
        const float x0 = bounds.x + bounds.width * (static_cast<float>(index - 1) / static_cast<float>(values.size() - 1));
        const float x1 = bounds.x + bounds.width * (static_cast<float>(index) / static_cast<float>(values.size() - 1));
        const float y0 = bounds.y + bounds.height - bounds.height * ((values[index - 1] - minValue) / range);
        const float y1 = bounds.y + bounds.height - bounds.height * ((values[index] - minValue) / range);
        DrawLineEx({x0, y0}, {x1, y1}, 2.0f, color);
    }
}

void PushHistory(std::vector<float>& values, const float value)
{
    values.push_back(value);
    if (values.size() > 180) {
        values.erase(values.begin());
    }
}

float ValueAt(const std::vector<float>& values, const std::size_t index)
{
    return index < values.size() ? values[index] : 0.0f;
}

void DrawHistogram(const std::vector<float>& values, const Rectangle bounds, const Color color)
{
    DrawRectangleRec(bounds, {24, 26, 34, 255});
    DrawRectangleLinesEx(bounds, 1.0f, {55, 60, 72, 255});
    if (values.empty()) {
        return;
    }

    constexpr int kBins = 18;
    int bins[kBins]{};
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    const float minValue = *minIt;
    const float maxValue = *maxIt;
    const float range = std::max(maxValue - minValue, 1.0e-5f);
    for (const float value : values) {
        const int bin = std::clamp(static_cast<int>(((value - minValue) / range) * static_cast<float>(kBins - 1)), 0, kBins - 1);
        ++bins[bin];
    }

    const int maxBin = std::max(1, *std::max_element(std::begin(bins), std::end(bins)));
    const float binWidth = bounds.width / static_cast<float>(kBins);
    for (int index = 0; index < kBins; ++index) {
        const float height = bounds.height * (static_cast<float>(bins[index]) / static_cast<float>(maxBin));
        DrawRectangleRec({bounds.x + (binWidth * static_cast<float>(index)),
                          bounds.y + bounds.height - height,
                          std::max(1.0f, binWidth - 2.0f),
                          height},
                         color);
    }
}

} // namespace

int RunNeuralNetworkWindow(const std::filesystem::path& snapshotPath)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1100, 760, "AI Neural Network Visualizer");
    SetTargetFPS(60);

    sim::ai::AiDebugSnapshot snapshot;
    int selectedLayer = -1;
    int selectedIndex = -1;
    while (!WindowShouldClose()) {
        sim::ai::ReadAiDebugSnapshot(snapshotPath, snapshot);
        BeginDrawing();
        ClearBackground(kBg);

        DrawText("Neural Network Brain Scanner", 24, 18, 24, kText);
        DrawText("Actor head: action logits/probabilities    Critic head: state value", 24, 48, 16, kMuted);

        const int inputCount = static_cast<int>(std::max<std::size_t>(snapshot.normalizedObservation.size(), 1));
        const int hiddenCount = static_cast<int>(std::max<std::size_t>(snapshot.hiddenActivations.size(), 1));
        const int outputCount = static_cast<int>(std::max<std::size_t>(snapshot.actionProbabilities.size(), 1));
        const float inputX = 120.0f;
        const float hiddenX = 500.0f;
        const float outputX = 880.0f;
        const float top = 105.0f;
        const float height = 470.0f;

        std::vector<Vector2> inputs;
        std::vector<Vector2> hidden;
        std::vector<Vector2> outputs;
        for (int i = 0; i < inputCount; ++i) {
            inputs.push_back({inputX, top + height * (static_cast<float>(i) + 0.5f) / static_cast<float>(inputCount)});
        }
        for (int i = 0; i < hiddenCount; ++i) {
            hidden.push_back({hiddenX, top + height * (static_cast<float>(i) + 0.5f) / static_cast<float>(hiddenCount)});
        }
        for (int i = 0; i < outputCount; ++i) {
            outputs.push_back({outputX, top + height * (static_cast<float>(i) + 0.5f) / static_cast<float>(outputCount)});
        }

        const std::vector<float>& params = snapshot.networkParameters;
        std::size_t paramIndex = 0;
        for (const Vector2& from : inputs) {
            for (const Vector2& to : hidden) {
                const float weight = paramIndex < params.size() ? params[paramIndex++] : 0.0f;
                const float mag = Clamp01(std::fabs(weight) / std::max(snapshot.weightStats.meanAbsWeight * 4.0f, 0.001f));
                Color c = weight >= 0.0f ? kGood : kBad;
                c.a = static_cast<unsigned char>(45 + mag * 180.0f);
                DrawLineEx(from, to, 0.5f + mag * 3.5f, c);
            }
        }
        for (const Vector2& from : hidden) {
            for (const Vector2& to : outputs) {
                const float weight = paramIndex < params.size() ? params[paramIndex++] : 0.0f;
                const float mag = Clamp01(std::fabs(weight) / std::max(snapshot.weightStats.meanAbsWeight * 4.0f, 0.001f));
                Color c = weight >= 0.0f ? kGood : kBad;
                c.a = static_cast<unsigned char>(45 + mag * 180.0f);
                DrawLineEx(from, to, 0.5f + mag * 3.5f, c);
            }
        }

        const Vector2 mouse = GetMousePosition();
        const bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        auto drawNeuron = [&](const Vector2 p, const float activation, const char* label) {
            const float brightness = Clamp01((activation + 1.0f) * 0.5f);
            const Color fill{static_cast<unsigned char>(45 + brightness * 130.0f),
                             static_cast<unsigned char>(75 + brightness * 150.0f),
                             static_cast<unsigned char>(95 + brightness * 130.0f),
                             255};
            DrawCircleV(p, 12.0f, fill);
            DrawCircleLinesV(p, 12.0f, kText);
            if (CheckCollisionPointCircle(mouse, p, 14.0f)) {
                char buffer[128];
                std::snprintf(buffer, sizeof(buffer), "%s activation %.3f", label, activation);
                DrawRectangle(static_cast<int>(mouse.x + 12), static_cast<int>(mouse.y + 12), MeasureText(buffer, 16) + 12, 24, kPanel);
                DrawText(buffer, static_cast<int>(mouse.x + 18), static_cast<int>(mouse.y + 16), 16, kText);
            }
        };

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const float v = i < snapshot.normalizedObservation.size() ? snapshot.normalizedObservation[i] : 0.0f;
            drawNeuron(inputs[i], v, "Input");
            if (clicked && CheckCollisionPointCircle(mouse, inputs[i], 14.0f)) {
                selectedLayer = 0;
                selectedIndex = static_cast<int>(i);
            }
        }
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            const float v = i < snapshot.hiddenActivations.size() ? snapshot.hiddenActivations[i] : 0.0f;
            drawNeuron(hidden[i], v, "Hidden");
            if (clicked && CheckCollisionPointCircle(mouse, hidden[i], 14.0f)) {
                selectedLayer = 1;
                selectedIndex = static_cast<int>(i);
            }
        }
        int selected = 0;
        if (!snapshot.actionProbabilities.empty()) {
            selected = static_cast<int>(std::distance(snapshot.actionProbabilities.begin(),
                                                      std::max_element(snapshot.actionProbabilities.begin(), snapshot.actionProbabilities.end())));
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const float p = i < snapshot.actionProbabilities.size() ? snapshot.actionProbabilities[i] : 0.0f;
            drawNeuron(outputs[i], p, "Action");
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "A%zu  %.2f", i, p);
            DrawText(buffer, static_cast<int>(outputs[i].x + 22), static_cast<int>(outputs[i].y - 8), 16,
                     static_cast<int>(i) == selected ? kWarn : kText);
            if (clicked && CheckCollisionPointCircle(mouse, outputs[i], 14.0f)) {
                selectedLayer = 2;
                selectedIndex = static_cast<int>(i);
            }
        }

        const float flowPhase = std::fmod(static_cast<float>(GetTime()) * 0.75f, 1.0f);
        if (!inputs.empty() && !hidden.empty() && !outputs.empty()) {
            const Vector2 a = inputs[static_cast<std::size_t>(snapshot.simulationTick % inputs.size())];
            const Vector2 b = hidden[static_cast<std::size_t>(snapshot.simulationTick % hidden.size())];
            const Vector2 c = outputs[static_cast<std::size_t>(std::max(snapshot.lastAction, 0)) % outputs.size()];
            const Vector2 flow = flowPhase < 0.5f
                                     ? Vector2{a.x + ((b.x - a.x) * flowPhase * 2.0f),
                                               a.y + ((b.y - a.y) * flowPhase * 2.0f)}
                                     : Vector2{b.x + ((c.x - b.x) * ((flowPhase - 0.5f) * 2.0f)),
                                               b.y + ((c.y - b.y) * ((flowPhase - 0.5f) * 2.0f))};
            DrawCircleV(flow, 5.0f, kWarn);
        }

        char info[256];
        std::snprintf(info, sizeof(info), "Critic value %.3f | TD error %.3f | Selected action %d",
                      snapshot.lastValue, snapshot.lastTdError, snapshot.lastAction);
        DrawText(info, 24, 650, 18, kAccent);
        std::snprintf(info, sizeof(info), "Weights min %.3f max %.3f mean|w| %.3f L2 %.3f",
                      snapshot.weightStats.minWeight, snapshot.weightStats.maxWeight,
                      snapshot.weightStats.meanAbsWeight, snapshot.weightStats.l2Norm);
        DrawText(info, 24, 680, 18, kMuted);

        DrawRectangle(720, 610, 350, 120, kPanel);
        DrawText("Parameter Inspector", 738, 626, 18, kText);
        if (selectedLayer >= 0 && selectedIndex >= 0) {
            const std::size_t inputHiddenCount = snapshot.normalizedObservation.size() * snapshot.hiddenActivations.size();
            const std::size_t hiddenBiasOffset = inputHiddenCount;
            const std::size_t hiddenOutputOffset = hiddenBiasOffset + snapshot.hiddenActivations.size();
            const std::size_t outputBiasOffset = hiddenOutputOffset +
                                                 (snapshot.hiddenActivations.size() * snapshot.actionProbabilities.size());
            const char* layerName = selectedLayer == 0 ? "Input" : selectedLayer == 1 ? "Hidden" : "Output";
            std::snprintf(info, sizeof(info), "%s neuron %d", layerName, selectedIndex);
            DrawText(info, 738, 654, 16, kAccent);
            if (selectedLayer == 1) {
                const float bias = ValueAt(snapshot.networkParameters, hiddenBiasOffset + static_cast<std::size_t>(selectedIndex));
                std::snprintf(info, sizeof(info), "bias %.5f  activation %.5f", bias,
                              ValueAt(snapshot.hiddenActivations, static_cast<std::size_t>(selectedIndex)));
                DrawText(info, 738, 678, 16, kText);
            } else if (selectedLayer == 2) {
                const float bias = ValueAt(snapshot.networkParameters, outputBiasOffset + static_cast<std::size_t>(selectedIndex));
                std::snprintf(info, sizeof(info), "bias %.5f  logit %.5f  prob %.5f", bias,
                              ValueAt(snapshot.actorLogits, static_cast<std::size_t>(selectedIndex)),
                              ValueAt(snapshot.actionProbabilities, static_cast<std::size_t>(selectedIndex)));
                DrawText(info, 738, 678, 16, kText);
            } else {
                std::snprintf(info, sizeof(info), "raw %.5f  normalized %.5f",
                              ValueAt(snapshot.observation, static_cast<std::size_t>(selectedIndex)),
                              ValueAt(snapshot.normalizedObservation, static_cast<std::size_t>(selectedIndex)));
                DrawText(info, 738, 678, 16, kText);
            }
            std::snprintf(info, sizeof(info), "gradient norm %.5f  update norm %.5f",
                          snapshot.trainingState.lastGradientNorm,
                          snapshot.trainingState.lastParameterUpdateNorm);
            DrawText(info, 738, 704, 16, kMuted);
        } else {
            DrawText("Click a neuron to inspect activation, bias, and update stats.", 738, 654, 16, kMuted);
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}

int RunPpoDashboardWindow(const std::filesystem::path& snapshotPath)
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1180, 820, "PPO Training Dashboard");
    SetTargetFPS(30);

    sim::ai::AiDebugSnapshot snapshot;
    std::vector<float> rewardHistory;
    std::vector<float> policyLossHistory;
    std::vector<float> valueLossHistory;
    std::vector<float> entropyHistory;
    std::vector<float> advantageHistory;
    std::vector<float> learningRateHistory;
    std::vector<float> valueEstimateHistory;
    std::vector<float> gradientNormHistory;
    std::vector<float> klHistory;
    std::vector<float> clipHistory;
    std::vector<float> weightNormHistory;
    std::vector<float> parameterDriftHistory;
    std::deque<std::string> decisionHistory;
    std::vector<float> previousParameters;
    std::uint64_t lastTick = 0;

    while (!WindowShouldClose()) {
        if (sim::ai::ReadAiDebugSnapshot(snapshotPath, snapshot) && snapshot.simulationTick != lastTick) {
            lastTick = snapshot.simulationTick;
            float drift = 0.0f;
            if (previousParameters.size() == snapshot.networkParameters.size()) {
                for (std::size_t index = 0; index < snapshot.networkParameters.size(); ++index) {
                    const float delta = snapshot.networkParameters[index] - previousParameters[index];
                    drift += delta * delta;
                }
                drift = std::sqrt(drift);
                snapshot.weightStats.parameterDrift = drift;
            }
            previousParameters = snapshot.networkParameters;
            PushHistory(rewardHistory, snapshot.lastReward);
            PushHistory(policyLossHistory, snapshot.trainingState.lastPolicyLoss);
            PushHistory(valueLossHistory, snapshot.trainingState.lastValueLoss);
            PushHistory(entropyHistory, snapshot.trainingState.lastEntropy);
            PushHistory(advantageHistory, snapshot.trainingState.lastMeanAdvantage);
            PushHistory(learningRateHistory, snapshot.learningRate);
            PushHistory(valueEstimateHistory, snapshot.lastValue);
            PushHistory(gradientNormHistory, snapshot.trainingState.lastGradientNorm);
            PushHistory(klHistory, snapshot.trainingState.lastKlDivergence);
            PushHistory(clipHistory, snapshot.trainingState.lastClipFraction);
            PushHistory(weightNormHistory, snapshot.weightStats.l2Norm);
            PushHistory(parameterDriftHistory, drift);

            char decision[256];
            std::snprintf(decision, sizeof(decision), "t=%llu obs=%zu action=%d reward=%.3f value=%.3f adv=%.3f return=%.3f",
                          static_cast<unsigned long long>(snapshot.simulationTick),
                          snapshot.observation.size(),
                          snapshot.lastAction,
                          snapshot.lastReward,
                          snapshot.lastValue,
                          snapshot.lastAdvantage,
                          snapshot.lastReturn);
            decisionHistory.emplace_front(decision);
            while (decisionHistory.size() > 9) {
                decisionHistory.pop_back();
            }
        }

        BeginDrawing();
        ClearBackground(kBg);
        DrawText("PPO Reinforcement Learning Dashboard", 24, 18, 24, kText);
        DrawText("Live training state from the running simulation process", 24, 48, 16, kMuted);

        DrawRectangle(20, 82, 390, 330, kPanel);
        int y = 102;
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.simulationTick));
        DrawMetric("Simulation Tick", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%.1f", snapshot.fps);
        DrawMetric("FPS", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.trainingState.trainingStepCount));
        DrawMetric("Total Timesteps", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.trainingState.policyUpdateCount));
        DrawMetric("PPO Updates", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%zu / %zu", snapshot.rolloutSize, snapshot.rolloutCapacity);
        DrawMetric("Rollout Buffer", buffer, 38, y, kAccent);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastReward);
        DrawMetric("Current Reward", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.totalReward);
        DrawMetric("Cumulative Reward", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastTdError);
        DrawMetric("TD Error", buffer, 38, y, snapshot.lastTdError >= 0.0f ? kGood : kBad);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastMeanAdvantage);
        DrawMetric("GAE Advantage", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastEntropy);
        DrawMetric("Entropy", buffer, 38, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastLogProbability);
        DrawMetric("Log Probability", buffer, 38, y);

        DrawRectangle(430, 82, 330, 330, kPanel);
        y = 102;
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastPolicyLoss);
        DrawMetric("Policy Loss", buffer, 448, y, kWarn);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastValueLoss);
        DrawMetric("Value Loss", buffer, 448, y, kWarn);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastTotalLoss);
        DrawMetric("Total PPO Loss", buffer, 448, y, kBad);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastEntropy * 0.01f);
        DrawMetric("Entropy Bonus", buffer, 448, y, kGood);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.learningRate);
        DrawMetric("Learning Rate", buffer, 448, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastClipFraction);
        DrawMetric("Clip Fraction", buffer, 448, y);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastKlDivergence);
        DrawMetric("KL Divergence", buffer, 448, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastExplainedVariance);
        DrawMetric("Explained Var", buffer, 448, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastValue);
        DrawMetric("Value Estimate", buffer, 448, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastNextValue);
        DrawMetric("Next Value", buffer, 448, y);

        DrawRectangle(780, 82, 370, 330, kPanel);
        DrawText("Action Distribution", 798, 102, 18, kText);
        for (std::size_t i = 0; i < snapshot.actionProbabilities.size(); ++i) {
            const float p = snapshot.actionProbabilities[i];
            const int barW = static_cast<int>(p * 270.0f);
            const int barY = 138 + static_cast<int>(i) * 42;
            DrawText(TextFormat("A%zu", i), 798, barY, 16, kMuted);
            DrawRectangle(840, barY, 280, 20, {42, 45, 56, 255});
            DrawRectangle(840, barY, barW, 20, i == static_cast<std::size_t>(snapshot.lastAction) ? kWarn : kAccent);
            DrawText(TextFormat("%.2f", p), 1125, barY, 16, kText);
        }
        y = 314;
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastGradientNorm);
        DrawMetric("Gradient Norm", buffer, 798, y);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastParameterUpdateNorm);
        DrawMetric("Update Norm", buffer, 798, y);
        std::snprintf(buffer, sizeof(buffer), "%zu / %zu", snapshot.trainingState.lastPpoEpoch,
                      snapshot.trainingState.lastPpoMinibatch);
        DrawMetric("Epoch / Batch", buffer, 798, y);
        std::snprintf(buffer, sizeof(buffer), "%s", snapshot.trainingState.learnedPolicyEnabled ? "Learned" : "FSM Fallback");
        DrawMetric("Action Source", buffer, 798, y, snapshot.trainingState.learnedPolicyEnabled ? kGood : kAccent);

        DrawText("Reward", 24, 440, 18, kText);
        DrawGraph(rewardHistory, {24, 466, 170, 95}, kGood);
        DrawText("Policy Loss", 314, 440, 18, kText);
        DrawGraph(policyLossHistory, {214, 466, 170, 95}, kWarn);
        DrawText("Value Loss", 604, 440, 18, kText);
        DrawGraph(valueLossHistory, {404, 466, 170, 95}, kBad);
        DrawText("Entropy", 894, 440, 18, kText);
        DrawGraph(entropyHistory, {594, 466, 170, 95}, kAccent);
        DrawText("Advantage", 784, 440, 18, kText);
        DrawGraph(advantageHistory, {784, 466, 170, 95}, kAccent);
        DrawText("Value", 974, 440, 18, kText);
        DrawGraph(valueEstimateHistory, {974, 466, 170, 95}, kGood);

        DrawText("Learning Rate", 24, 580, 18, kText);
        DrawGraph(learningRateHistory, {24, 606, 170, 95}, kAccent);
        DrawText("Gradient", 214, 580, 18, kText);
        DrawGraph(gradientNormHistory, {214, 606, 170, 95}, kWarn);
        DrawText("KL", 404, 580, 18, kText);
        DrawGraph(klHistory, {404, 606, 170, 95}, kBad);
        DrawText("Clip", 594, 580, 18, kText);
        DrawGraph(clipHistory, {594, 606, 170, 95}, kAccent);
        DrawText("Weight Norm", 784, 580, 18, kText);
        DrawGraph(weightNormHistory, {784, 606, 170, 95}, kGood);
        DrawText("Param Drift", 974, 580, 18, kText);
        DrawGraph(parameterDriftHistory, {974, 606, 170, 95}, kWarn);

        DrawRectangle(20, 718, 390, 82, kPanel);
        DrawText("Weight Histogram", 38, 734, 18, kText);
        DrawHistogram(snapshot.networkParameters, {38, 760, 350, 28}, kAccent);

        DrawRectangle(430, 718, 720, 82, kPanel);
        DrawText("Decision History", 448, 734, 18, kText);
        int historyY = 760;
        for (const std::string& decision : decisionHistory) {
            DrawText(decision.c_str(), 448, historyY, 14, kMuted);
            historyY += 18;
            if (historyY > 790) {
                break;
            }
        }

        DrawText("Training Flow: Observation -> Forward Pass -> Action -> Environment Step -> Reward -> Trajectory -> GAE -> PPO Update -> Checkpoint",
                 24, 706, 14, kMuted);

        EndDrawing();
    }

    CloseWindow();
    return 0;
}

} // namespace sim::visualization
