#include "sim/visualization/AiDebugWindows.hpp"

#include "sim/ai/AiDebugSnapshot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iterator>
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
constexpr float kMinZoom = 0.35f;
constexpr float kMaxZoom = 2.50f;

struct CanvasNavigation {
    Vector2 target{0.0f, 0.0f};
    Vector2 velocity{0.0f, 0.0f};
    bool draggingCanvas{false};
    bool draggingHorizontalBar{false};
    bool draggingVerticalBar{false};
    Vector2 lastMouse{0.0f, 0.0f};
    float zoom{1.0f};
};

float Clamp01(const float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float MaxTargetX(const Vector2 canvasSize, const Vector2 viewportSize, const float zoom)
{
    return std::max(0.0f, canvasSize.x - (viewportSize.x / std::max(zoom, 0.001f)));
}

float MaxTargetY(const Vector2 canvasSize, const Vector2 viewportSize, const float zoom)
{
    return std::max(0.0f, canvasSize.y - (viewportSize.y / std::max(zoom, 0.001f)));
}

void ClampNavigation(CanvasNavigation& nav, const Vector2 canvasSize, const Vector2 viewportSize)
{
    nav.zoom = std::clamp(nav.zoom, kMinZoom, kMaxZoom);
    nav.target.x = std::clamp(nav.target.x, 0.0f, MaxTargetX(canvasSize, viewportSize, nav.zoom));
    nav.target.y = std::clamp(nav.target.y, 0.0f, MaxTargetY(canvasSize, viewportSize, nav.zoom));
}

Rectangle HorizontalScrollbarThumb(const CanvasNavigation& nav, const Vector2 canvasSize, const Vector2 viewportSize)
{
    const float barHeight = 12.0f;
    const float visibleWorldWidth = viewportSize.x / std::max(nav.zoom, 0.001f);
    const float ratio = std::clamp(visibleWorldWidth / std::max(canvasSize.x, 1.0f), 0.0f, 1.0f);
    const float thumbWidth = std::max(36.0f, (viewportSize.x - 14.0f) * ratio);
    const float travel = std::max(1.0f, viewportSize.x - 14.0f - thumbWidth);
    const float maxTarget = MaxTargetX(canvasSize, viewportSize, nav.zoom);
    const float x = maxTarget > 0.0f ? (nav.target.x / maxTarget) * travel : 0.0f;
    return {x, viewportSize.y - barHeight, thumbWidth, barHeight};
}

Rectangle VerticalScrollbarThumb(const CanvasNavigation& nav, const Vector2 canvasSize, const Vector2 viewportSize)
{
    const float barWidth = 12.0f;
    const float visibleWorldHeight = viewportSize.y / std::max(nav.zoom, 0.001f);
    const float ratio = std::clamp(visibleWorldHeight / std::max(canvasSize.y, 1.0f), 0.0f, 1.0f);
    const float thumbHeight = std::max(36.0f, (viewportSize.y - 14.0f) * ratio);
    const float travel = std::max(1.0f, viewportSize.y - 14.0f - thumbHeight);
    const float maxTarget = MaxTargetY(canvasSize, viewportSize, nav.zoom);
    const float y = maxTarget > 0.0f ? (nav.target.y / maxTarget) * travel : 0.0f;
    return {viewportSize.x - barWidth, y, barWidth, thumbHeight};
}

void UpdateNavigation(CanvasNavigation& nav, const Vector2 canvasSize, const bool allowLeftDrag)
{
    const Vector2 viewport{static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())};
    const Vector2 mouse = GetMousePosition();
    const float dt = std::min(GetFrameTime(), 1.0f / 20.0f);
    const bool ctrlDown = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);
    const Rectangle horizontalThumb = HorizontalScrollbarThumb(nav, canvasSize, viewport);
    const Rectangle verticalThumb = VerticalScrollbarThumb(nav, canvasSize, viewport);

    if (ctrlDown) {
        const float wheel = GetMouseWheelMove();
        if (std::fabs(wheel) > 0.01f) {
            const Vector2 worldBefore{mouse.x / nav.zoom + nav.target.x, mouse.y / nav.zoom + nav.target.y};
            const float zoomFactor = 1.0f + (wheel * 0.10f);
            nav.zoom = std::clamp(nav.zoom * std::max(0.2f, zoomFactor), kMinZoom, kMaxZoom);
            nav.target = {worldBefore.x - (mouse.x / nav.zoom), worldBefore.y - (mouse.y / nav.zoom)};
        }
    } else {
        const Vector2 wheel = GetMouseWheelMoveV();
        const float verticalWheel = std::fabs(wheel.y) > 0.01f ? wheel.y : GetMouseWheelMove();
        nav.velocity.x -= wheel.x * 120.0f / std::max(nav.zoom, 0.001f);
        nav.velocity.y -= verticalWheel * 120.0f / std::max(nav.zoom, 0.001f);
    }

    const float keyStep = 720.0f * dt / std::max(nav.zoom, 0.001f);
    if (IsKeyDown(KEY_RIGHT)) nav.velocity.x += keyStep * 8.0f;
    if (IsKeyDown(KEY_LEFT)) nav.velocity.x -= keyStep * 8.0f;
    if (IsKeyDown(KEY_DOWN)) nav.velocity.y += keyStep * 8.0f;
    if (IsKeyDown(KEY_UP)) nav.velocity.y -= keyStep * 8.0f;
    if (IsKeyPressed(KEY_PAGE_DOWN)) nav.velocity.y += viewport.y / std::max(nav.zoom, 0.001f);
    if (IsKeyPressed(KEY_PAGE_UP)) nav.velocity.y -= viewport.y / std::max(nav.zoom, 0.001f);
    if (IsKeyPressed(KEY_HOME)) nav.target = {0.0f, 0.0f};
    if (IsKeyPressed(KEY_END)) {
        nav.target = {MaxTargetX(canvasSize, viewport, nav.zoom), MaxTargetY(canvasSize, viewport, nav.zoom)};
    }

    const bool leftDragAllowed = allowLeftDrag && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
                                 !CheckCollisionPointRec(mouse, horizontalThumb) &&
                                 !CheckCollisionPointRec(mouse, verticalThumb);
    if (IsMouseButtonPressed(MOUSE_BUTTON_MIDDLE) || IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || leftDragAllowed) {
        nav.draggingCanvas = true;
        nav.lastMouse = mouse;
        nav.velocity = {0.0f, 0.0f};
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, horizontalThumb)) {
        nav.draggingHorizontalBar = true;
        nav.lastMouse = mouse;
        nav.velocity = {0.0f, 0.0f};
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(mouse, verticalThumb)) {
        nav.draggingVerticalBar = true;
        nav.lastMouse = mouse;
        nav.velocity = {0.0f, 0.0f};
    }

    if (nav.draggingCanvas) {
        const Vector2 delta{mouse.x - nav.lastMouse.x, mouse.y - nav.lastMouse.y};
        nav.target.x -= delta.x / std::max(nav.zoom, 0.001f);
        nav.target.y -= delta.y / std::max(nav.zoom, 0.001f);
        nav.lastMouse = mouse;
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && !IsMouseButtonDown(MOUSE_BUTTON_RIGHT) && !IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            nav.draggingCanvas = false;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_RIGHT) || IsMouseButtonReleased(MOUSE_BUTTON_MIDDLE)) {
            nav.draggingCanvas = false;
        }
    }

    if (nav.draggingHorizontalBar) {
        const float travel = std::max(1.0f, viewport.x - 14.0f - horizontalThumb.width);
        nav.target.x += ((mouse.x - nav.lastMouse.x) / travel) * MaxTargetX(canvasSize, viewport, nav.zoom);
        nav.lastMouse = mouse;
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) nav.draggingHorizontalBar = false;
    }
    if (nav.draggingVerticalBar) {
        const float travel = std::max(1.0f, viewport.y - 14.0f - verticalThumb.height);
        nav.target.y += ((mouse.y - nav.lastMouse.y) / travel) * MaxTargetY(canvasSize, viewport, nav.zoom);
        nav.lastMouse = mouse;
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) nav.draggingVerticalBar = false;
    }

    nav.target.x += nav.velocity.x * dt;
    nav.target.y += nav.velocity.y * dt;
    nav.velocity.x *= std::pow(0.08f, dt);
    nav.velocity.y *= std::pow(0.08f, dt);
    if (std::fabs(nav.velocity.x) < 0.01f) nav.velocity.x = 0.0f;
    if (std::fabs(nav.velocity.y) < 0.01f) nav.velocity.y = 0.0f;
    ClampNavigation(nav, canvasSize, viewport);
}

Camera2D NavigationCamera(const CanvasNavigation& nav)
{
    return {.offset = {0.0f, 0.0f}, .target = nav.target, .rotation = 0.0f, .zoom = nav.zoom};
}

void DrawNavigationOverlay(const CanvasNavigation& nav, const Vector2 canvasSize)
{
    const Vector2 viewport{static_cast<float>(GetScreenWidth()), static_cast<float>(GetScreenHeight())};
    DrawRectangle(0, GetScreenHeight() - 12, GetScreenWidth() - 12, 12, {10, 12, 18, 220});
    DrawRectangle(GetScreenWidth() - 12, 0, 12, GetScreenHeight() - 12, {10, 12, 18, 220});
    DrawRectangleRec(HorizontalScrollbarThumb(nav, canvasSize, viewport), {100, 112, 132, 230});
    DrawRectangleRec(VerticalScrollbarThumb(nav, canvasSize, viewport), {100, 112, 132, 230});
    DrawRectangle(GetScreenWidth() - 12, GetScreenHeight() - 12, 12, 12, {10, 12, 18, 255});
    DrawText(TextFormat("Zoom %.0f%%", nav.zoom * 100.0f), 16, GetScreenHeight() - 34, 16, kMuted);
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
    CanvasNavigation nav;
    int selectedLayer = -1;
    int selectedIndex = -1;
    while (!WindowShouldClose()) {
        sim::ai::ReadAiDebugSnapshot(snapshotPath, snapshot);
        const int inputCount = static_cast<int>(std::max<std::size_t>(snapshot.normalizedObservation.size(), 1));
        const int hiddenCount = static_cast<int>(std::max<std::size_t>(snapshot.hiddenActivations.size(), 1));
        const int outputCount = static_cast<int>(std::max<std::size_t>(snapshot.actionProbabilities.size(), 1));
        const int largestLayer = std::max({inputCount, hiddenCount, outputCount});
        const Vector2 canvasSize{std::max(1220.0f, 360.0f + static_cast<float>(largestLayer) * 18.0f),
                                 std::max(860.0f, 210.0f + static_cast<float>(largestLayer) * 64.0f)};
        UpdateNavigation(nav, canvasSize, false);

        BeginDrawing();
        ClearBackground(kBg);
        BeginMode2D(NavigationCamera(nav));

        DrawText("Neural Network Brain Scanner", 24, 18, 24, kText);
        DrawText("Actor head: action logits/probabilities    Critic head: state value", 24, 48, 16, kMuted);

        const float inputX = 140.0f;
        const float hiddenX = canvasSize.x * 0.50f;
        const float outputX = canvasSize.x - 250.0f;
        const float top = 105.0f;
        const float height = canvasSize.y - 310.0f;

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
        const Rectangle visibleWorld{nav.target.x - 80.0f,
                                     nav.target.y - 80.0f,
                                     static_cast<float>(GetScreenWidth()) / nav.zoom + 160.0f,
                                     static_cast<float>(GetScreenHeight()) / nav.zoom + 160.0f};
        auto isVisible = [&](const Vector2 point) {
            return CheckCollisionPointRec(point, visibleWorld);
        };
        for (const Vector2& from : inputs) {
            for (const Vector2& to : hidden) {
                const float weight = paramIndex < params.size() ? params[paramIndex++] : 0.0f;
                if (!isVisible(from) && !isVisible(to)) {
                    continue;
                }
                const float mag = Clamp01(std::fabs(weight) / std::max(snapshot.weightStats.meanAbsWeight * 4.0f, 0.001f));
                Color c = weight >= 0.0f ? kGood : kBad;
                c.a = static_cast<unsigned char>(45 + mag * 180.0f);
                DrawLineEx(from, to, 0.5f + mag * 3.5f, c);
            }
        }
        for (const Vector2& from : hidden) {
            for (const Vector2& to : outputs) {
                const float weight = paramIndex < params.size() ? params[paramIndex++] : 0.0f;
                if (!isVisible(from) && !isVisible(to)) {
                    continue;
                }
                const float mag = Clamp01(std::fabs(weight) / std::max(snapshot.weightStats.meanAbsWeight * 4.0f, 0.001f));
                Color c = weight >= 0.0f ? kGood : kBad;
                c.a = static_cast<unsigned char>(45 + mag * 180.0f);
                DrawLineEx(from, to, 0.5f + mag * 3.5f, c);
            }
        }

        const Vector2 mouse = GetScreenToWorld2D(GetMousePosition(), NavigationCamera(nav));
        const bool clicked = IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
        auto drawNeuron = [&](const Vector2 p, const float activation, const char* label) {
            if (!isVisible(p)) {
                return;
            }
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

        EndMode2D();
        DrawNavigationOverlay(nav, canvasSize);
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
    CanvasNavigation nav;
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
            while (decisionHistory.size() > 30) {
                decisionHistory.pop_back();
            }
        }

        const float viewportWorldWidth = static_cast<float>(GetScreenWidth()) / std::max(nav.zoom, 0.001f);
        const float contentWidth = std::max(1120.0f, viewportWorldWidth - 48.0f);
        const float graphWidth = std::max(260.0f, (contentWidth - 72.0f) / 3.0f);
        const float graphHeight = 130.0f;
        const float historyHeight = 80.0f + static_cast<float>(decisionHistory.size()) * 20.0f;
        const Vector2 canvasSize{contentWidth + 48.0f, 1240.0f + historyHeight};
        UpdateNavigation(nav, canvasSize, true);

        BeginDrawing();
        ClearBackground(kBg);
        BeginMode2D(NavigationCamera(nav));

        DrawText("PPO Reinforcement Learning Dashboard", 24, 18, 24, kText);
        DrawText("Live training state from the running simulation process", 24, 48, 16, kMuted);

        const float gap = 20.0f;
        const float panelTop = 82.0f;
        const float panelWidth = (contentWidth - (gap * 2.0f)) / 3.0f;
        DrawRectangleRec({24.0f, panelTop, panelWidth, 330.0f}, kPanel);
        int y = 102;
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.simulationTick));
        DrawMetric("Simulation Tick", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%.1f", snapshot.fps);
        DrawMetric("FPS", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.trainingState.trainingStepCount));
        DrawMetric("Total Timesteps", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(snapshot.trainingState.policyUpdateCount));
        DrawMetric("PPO Updates", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%zu / %zu", snapshot.rolloutSize, snapshot.rolloutCapacity);
        DrawMetric("Rollout Buffer", buffer, 42, y, kAccent);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastReward);
        DrawMetric("Current Reward", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.totalReward);
        DrawMetric("Cumulative Reward", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastTdError);
        DrawMetric("TD Error", buffer, 42, y, snapshot.lastTdError >= 0.0f ? kGood : kBad);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastMeanAdvantage);
        DrawMetric("GAE Advantage", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastEntropy);
        DrawMetric("Entropy", buffer, 42, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastLogProbability);
        DrawMetric("Log Probability", buffer, 42, y);

        const int middleX = static_cast<int>(24.0f + panelWidth + gap);
        DrawRectangleRec({static_cast<float>(middleX), panelTop, panelWidth, 330.0f}, kPanel);
        y = 102;
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastPolicyLoss);
        DrawMetric("Policy Loss", buffer, middleX + 18, y, kWarn);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastValueLoss);
        DrawMetric("Value Loss", buffer, middleX + 18, y, kWarn);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastTotalLoss);
        DrawMetric("Total PPO Loss", buffer, middleX + 18, y, kBad);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastEntropy * 0.01f);
        DrawMetric("Entropy Bonus", buffer, middleX + 18, y, kGood);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.learningRate);
        DrawMetric("Learning Rate", buffer, middleX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastClipFraction);
        DrawMetric("Clip Fraction", buffer, middleX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastKlDivergence);
        DrawMetric("KL Divergence", buffer, middleX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastExplainedVariance);
        DrawMetric("Explained Var", buffer, middleX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastValue);
        DrawMetric("Value Estimate", buffer, middleX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.lastNextValue);
        DrawMetric("Next Value", buffer, middleX + 18, y);

        const int rightX = static_cast<int>(24.0f + (panelWidth + gap) * 2.0f);
        DrawRectangleRec({static_cast<float>(rightX), panelTop, panelWidth, 330.0f}, kPanel);
        DrawText("Action Distribution", rightX + 18, 102, 18, kText);
        const int actionBarX = rightX + 60;
        const int actionBarWidth = static_cast<int>(std::max(120.0f, panelWidth - 115.0f));
        for (std::size_t i = 0; i < snapshot.actionProbabilities.size(); ++i) {
            const float p = snapshot.actionProbabilities[i];
            const int barW = static_cast<int>(p * static_cast<float>(actionBarWidth));
            const int barY = 138 + static_cast<int>(i) * 42;
            DrawText(TextFormat("A%zu", i), rightX + 18, barY, 16, kMuted);
            DrawRectangle(actionBarX, barY, actionBarWidth, 20, {42, 45, 56, 255});
            DrawRectangle(actionBarX, barY, barW, 20, i == static_cast<std::size_t>(snapshot.lastAction) ? kWarn : kAccent);
            DrawText(TextFormat("%.2f", p), actionBarX + actionBarWidth + 8, barY, 16, kText);
        }
        y = 314;
        std::snprintf(buffer, sizeof(buffer), "%.3f", snapshot.trainingState.lastGradientNorm);
        DrawMetric("Gradient Norm", buffer, rightX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%.5f", snapshot.trainingState.lastParameterUpdateNorm);
        DrawMetric("Update Norm", buffer, rightX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%zu / %zu", snapshot.trainingState.lastPpoEpoch,
                      snapshot.trainingState.lastPpoMinibatch);
        DrawMetric("Epoch / Batch", buffer, rightX + 18, y);
        std::snprintf(buffer, sizeof(buffer), "%s", snapshot.trainingState.learnedPolicyEnabled ? "Learned" : "FSM Fallback");
        DrawMetric("Action Source", buffer, rightX + 18, y, snapshot.trainingState.learnedPolicyEnabled ? kGood : kAccent);

        const float graphTop = 450.0f;
        struct GraphSpec {
            const char* title;
            const std::vector<float>* values;
            Color color;
        };
        const GraphSpec graphs[] = {
            {"Reward", &rewardHistory, kGood},
            {"Policy Loss", &policyLossHistory, kWarn},
            {"Value Loss", &valueLossHistory, kBad},
            {"Entropy", &entropyHistory, kAccent},
            {"Advantage", &advantageHistory, kAccent},
            {"Value Estimate", &valueEstimateHistory, kGood},
            {"Learning Rate", &learningRateHistory, kAccent},
            {"Gradient Norm", &gradientNormHistory, kWarn},
            {"KL Divergence", &klHistory, kBad},
            {"Clip Fraction", &clipHistory, kAccent},
            {"Weight Norm", &weightNormHistory, kGood},
            {"Parameter Drift", &parameterDriftHistory, kWarn}
        };
        for (std::size_t index = 0; index < std::size(graphs); ++index) {
            const float column = static_cast<float>(index % 3);
            const float row = static_cast<float>(index / 3);
            const float graphX = 24.0f + column * (graphWidth + gap);
            const float titleY = graphTop + row * (graphHeight + 54.0f);
            DrawText(graphs[index].title, static_cast<int>(graphX), static_cast<int>(titleY), 18, kText);
            DrawGraph(*graphs[index].values, {graphX, titleY + 26.0f, graphWidth, graphHeight}, graphs[index].color);
        }

        const float histogramTop = graphTop + 4.0f * (graphHeight + 54.0f) + 12.0f;
        DrawRectangleRec({24.0f, histogramTop, contentWidth, 170.0f}, kPanel);
        DrawText("Weight Histogram and Parameter Evolution", 42, static_cast<int>(histogramTop + 18.0f), 18, kText);
        DrawHistogram(snapshot.networkParameters, {42.0f, histogramTop + 52.0f, contentWidth - 36.0f, 90.0f}, kAccent);
        DrawText(TextFormat("min %.4f   max %.4f   mean|w| %.4f   L2 %.4f   drift %.6f   max update %.6f",
                            snapshot.weightStats.minWeight,
                            snapshot.weightStats.maxWeight,
                            snapshot.weightStats.meanAbsWeight,
                            snapshot.weightStats.l2Norm,
                            snapshot.weightStats.parameterDrift,
                            snapshot.weightStats.maxParameterUpdate),
                 42, static_cast<int>(histogramTop + 146.0f), 16, kMuted);

        const float historyTop = histogramTop + 190.0f;
        DrawRectangleRec({24.0f, historyTop, contentWidth, historyHeight}, kPanel);
        DrawText("Decision History", 42, static_cast<int>(historyTop + 18.0f), 18, kText);
        int historyY = static_cast<int>(historyTop + 52.0f);
        for (const std::string& decision : decisionHistory) {
            DrawText(decision.c_str(), 42, historyY, 14, kMuted);
            historyY += 20;
        }

        DrawText("Training Flow: Observation -> Forward Pass -> Action -> Environment Step -> Reward -> Trajectory -> GAE -> PPO Update -> Checkpoint",
                 42, static_cast<int>(historyTop + historyHeight - 24.0f), 14, kMuted);

        EndMode2D();
        DrawNavigationOverlay(nav, canvasSize);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

} // namespace sim::visualization
