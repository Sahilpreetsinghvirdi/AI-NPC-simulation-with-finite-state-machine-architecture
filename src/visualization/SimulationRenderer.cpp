#include "sim/visualization/SimulationRenderer.hpp"

#include "sim/entities/NpcAction.hpp"
#include "sim/entities/NpcState.hpp"
#include "sim/entities/Player.hpp"
#include "sim/math/Vec2.hpp"

#include <cstdio>
#include <raylib.h>
#include <algorithm>

namespace sim::visualization {

namespace {

constexpr Color kBackgroundColor{24, 26, 32, 255};
constexpr Color kPanelColor{34, 37, 46, 255};
constexpr Color kGridColor{55, 60, 72, 255};
constexpr Color kPlayerColor{66, 135, 245, 255};
constexpr Color kPoliceColor{220, 80, 70, 255};
constexpr Color kTextPrimary{230, 233, 240, 255};
constexpr Color kTextMuted{150, 156, 170, 255};
constexpr Color kAccent{100, 210, 120, 255};
constexpr Color kInterceptLineColor{250, 210, 90, 180};
constexpr Color kVelocityColor{80, 220, 255, 220};
constexpr Color kDesiredHeadingColor{180, 120, 255, 210};
constexpr Color kSteeringColor{255, 120, 150, 210};
constexpr Color kPerceptionColor{120, 180, 255, 95};
constexpr Color kVisibleBoundaryColor{255, 120, 90, 230};
constexpr Color kMemoryBoundaryColor{255, 210, 90, 150};

void DrawEntity(const float screenX, const float screenY, const float radius, const Color fillColor,
                const Color outlineColor, const char* label)
{
    DrawCircleV({screenX, screenY}, radius, fillColor);
    DrawCircleLinesV({screenX, screenY}, radius, outlineColor);
    DrawText(label, static_cast<int>(screenX - 24.0f), static_cast<int>(screenY - radius - 22.0f), 16, kTextPrimary);
}

void DrawHealthBar(const float screenX, const float screenY, const float width, const float health,
                   const float maxHealth, const Color fillColor)
{
    const float ratio = std::clamp(health / maxHealth, 0.0f, 1.0f);
    const float barHeight = 6.0f;
    const float barX = screenX - (width * 0.5f);
    const float barY = screenY - 28.0f;

    DrawRectangle(static_cast<int>(barX), static_cast<int>(barY),
                  static_cast<int>(width), static_cast<int>(barHeight), {40, 42, 50, 255});
    DrawRectangle(static_cast<int>(barX), static_cast<int>(barY),
                  static_cast<int>(width * ratio), static_cast<int>(barHeight), fillColor);
    DrawRectangleLines(static_cast<int>(barX), static_cast<int>(barY),
                       static_cast<int>(width), static_cast<int>(barHeight), RAYWHITE);
}

} // namespace

SimulationRenderer::SimulationRenderer()
    : SimulationRenderer(Config{})
{
}

SimulationRenderer::SimulationRenderer(const Config config)
    : config_(config)
{
    InitWindow(config_.screenWidth, config_.screenHeight, "AiNpcSimulation — 2D Sandbox");
    SetTargetFPS(60);
}

SimulationRenderer::~SimulationRenderer()
{
    if (IsWindowReady()) {
        CloseWindow();
    }
}

bool SimulationRenderer::ShouldClose() const
{
    return WindowShouldClose();
}

float SimulationRenderer::GetFrameDeltaSeconds() const
{
    return GetFrameTime();
}

void SimulationRenderer::UpdateSimulation(Simulation& simulation, const float frameDeltaSeconds)
{
    simulation.UpdateRealtime(frameDeltaSeconds);

    if (simulation.IsPaused()) {
        return;
    }

    tickAccumulator_ += frameDeltaSeconds;

    const float tickDuration = 1.0f / simulation.GetTimer().GetTickRateHz();
    while (tickAccumulator_ >= tickDuration && !simulation.IsFinished()) {
        simulation.Tick();
        tickAccumulator_ -= tickDuration;
    }
}

void SimulationRenderer::Draw(const Simulation& simulation)
{
    BeginDrawing();
    ClearBackground(kBackgroundColor);

    DrawWorldPanel(simulation);
    DrawHudPanel(simulation);

    EndDrawing();
}

SimulationRenderer::ScreenPoint SimulationRenderer::WorldToScreen(const sim::math::Vec2 worldPosition) const
{
    const int worldWidthPx = config_.screenWidth - config_.hudWidth - (config_.worldMargin * 2);
    const int worldHeightPx = config_.screenHeight - (config_.worldMargin * 2);

    const float normalizedX = worldPosition.x / config_.worldWidth;
    const float normalizedY = worldPosition.y / config_.worldHeight;

    return {
        static_cast<float>(config_.worldMargin) + (normalizedX * worldWidthPx),
        static_cast<float>(config_.screenHeight - config_.worldMargin) - (normalizedY * worldHeightPx)
    };
}

void SimulationRenderer::DrawWorldPanel(const Simulation& simulation)
{
    const int worldRight = config_.screenWidth - config_.hudWidth;
    const Rectangle worldBounds{
        static_cast<float>(config_.worldMargin),
        static_cast<float>(config_.worldMargin),
        static_cast<float>(worldRight - config_.worldMargin),
        static_cast<float>(config_.screenHeight - (config_.worldMargin * 2))
    };

    DrawRectangleRec(worldBounds, kPanelColor);
    DrawRectangleLinesEx(worldBounds, 2.0f, kGridColor);

    const int gridLines = 6;
    for (int index = 1; index < gridLines; ++index) {
        const float t = static_cast<float>(index) / static_cast<float>(gridLines);
        const float x = worldBounds.x + (worldBounds.width * t);
        const float y = worldBounds.y + (worldBounds.height * t);
        DrawLineV({x, worldBounds.y}, {x, worldBounds.y + worldBounds.height}, kGridColor);
        DrawLineV({worldBounds.x, y}, {worldBounds.x + worldBounds.width, y}, kGridColor);
    }

    const auto& player = simulation.GetPlayer();
    const auto& policeManager = simulation.GetPoliceManager();

    const ScreenPoint hospitalScreen = WorldToScreen(simulation.GetHospitalPosition());
    DrawRectangle(static_cast<int>(hospitalScreen.x - 12.0f), static_cast<int>(hospitalScreen.y - 12.0f),
                  24, 24, {80, 200, 120, 180});
    DrawRectangleLines(static_cast<int>(hospitalScreen.x - 12.0f), static_cast<int>(hospitalScreen.y - 12.0f),
                       24, 24, RAYWHITE);
    DrawText("H", static_cast<int>(hospitalScreen.x - 6.0f), static_cast<int>(hospitalScreen.y - 10.0f), 18, RAYWHITE);

    const ScreenPoint playerScreen = WorldToScreen(player.GetPosition());
    const ScreenPoint velocityEnd = WorldToScreen(player.GetPosition() + (simulation.GetPlayerVelocity() * 1.2f));
    const ScreenPoint playerDesiredEnd = WorldToScreen(player.GetPosition() + (simulation.GetPlayerDesiredHeading() * 14.0f));
    const ScreenPoint playerSteeringEnd = WorldToScreen(player.GetPosition() + (simulation.GetPlayerSteeringVector() * 18.0f));

    const Color playerDrawColor = player.IsDead() ? Color{100, 100, 110, 255} : kPlayerColor;
    DrawCircleLines(static_cast<int>(playerScreen.x), static_cast<int>(playerScreen.y),
                    simulation.GetPlayerPerceptionRadius() *
                        ((config_.screenWidth - config_.hudWidth - (config_.worldMargin * 2)) / config_.worldWidth),
                    kPerceptionColor);
    DrawEntity(playerScreen.x, playerScreen.y, 10.0f, playerDrawColor, RAYWHITE, "Player");
    DrawLineEx({playerScreen.x, playerScreen.y}, {velocityEnd.x, velocityEnd.y}, 3.0f, kVelocityColor);
    DrawLineEx({playerScreen.x, playerScreen.y}, {playerDesiredEnd.x, playerDesiredEnd.y}, 2.0f, kDesiredHeadingColor);
    DrawLineEx({playerScreen.x, playerScreen.y}, {playerSteeringEnd.x, playerSteeringEnd.y}, 2.0f, kSteeringColor);
    DrawHealthBar(playerScreen.x, playerScreen.y, 40.0f, player.GetHealth(),
                  sim::entities::Player::kMaxHealth, kPlayerColor);

    for (const auto& boundary : simulation.GetPlayerKnownBoundaries()) {
        if (!boundary.remembered) {
            continue;
        }

        const ScreenPoint start = WorldToScreen(boundary.start);
        const ScreenPoint end = WorldToScreen(boundary.end);
        DrawLineEx({start.x, start.y}, {end.x, end.y}, boundary.visible ? 4.0f : 2.0f,
                   boundary.visible ? kVisibleBoundaryColor : kMemoryBoundaryColor);
    }

    std::size_t policeIndex = 0;
    for (const auto& police : policeManager.GetPoliceNpcs()) {
        const ScreenPoint policeScreen = WorldToScreen(police.GetPosition());
        const ScreenPoint targetScreen = WorldToScreen(police.GetCurrentTarget());
        const ScreenPoint policeDesiredEnd = WorldToScreen(police.GetPosition() + (police.GetDesiredHeading() * 12.0f));
        const ScreenPoint policeSteeringEnd = WorldToScreen(police.GetPosition() + (police.GetSteeringVector() * 16.0f));
        const ScreenPoint policePursuitEnd = WorldToScreen(police.GetPosition() + (police.GetPursuitVector() * 10.0f));
        DrawLineV({policeScreen.x, policeScreen.y}, {targetScreen.x, targetScreen.y}, kInterceptLineColor);
        DrawLineEx({policeScreen.x, policeScreen.y}, {policePursuitEnd.x, policePursuitEnd.y}, 2.0f, kVelocityColor);
        DrawLineEx({policeScreen.x, policeScreen.y}, {policeDesiredEnd.x, policeDesiredEnd.y}, 2.0f, kDesiredHeadingColor);
        DrawLineEx({policeScreen.x, policeScreen.y}, {policeSteeringEnd.x, policeSteeringEnd.y}, 2.0f, kSteeringColor);
        DrawCircleLines(static_cast<int>(targetScreen.x), static_cast<int>(targetScreen.y), 7.0f, kInterceptLineColor);
        DrawEntity(policeScreen.x, policeScreen.y, 10.0f, kPoliceColor, RAYWHITE, "Police");
        DrawText(policeManager.GetRoleName(policeIndex),
                 static_cast<int>(policeScreen.x - 34.0f),
                 static_cast<int>(policeScreen.y - 48.0f),
                 14,
                 kAccent);
        DrawHealthBar(policeScreen.x, policeScreen.y, 40.0f, police.GetHealth(),
                      sim::entities::PoliceNpc::kMaxHealth, kPoliceColor);
        ++policeIndex;
    }

    DrawText("World View", config_.worldMargin + 8, config_.worldMargin + 8, 18, kTextMuted);

    if (simulation.IsPlayerDead() || simulation.IsPaused()) {
        const char* deathText = "PLAYER DIED";
        const int fontSize = 48;
        const int textWidth = MeasureText(deathText, fontSize);
        const int centerX = static_cast<int>(worldBounds.x + (worldBounds.width - textWidth) * 0.5f);
        const int centerY = static_cast<int>(worldBounds.y + worldBounds.height * 0.5f - 24.0f);
        DrawText(deathText, centerX, centerY, fontSize, RED);

        char pauseBuffer[64];
        std::snprintf(pauseBuffer, sizeof(pauseBuffer), "Respawn in %.1fs", simulation.GetDeathPauseRemaining());
        const int pauseWidth = MeasureText(pauseBuffer, 24);
        DrawText(pauseBuffer, centerX + (textWidth - pauseWidth) / 2, centerY + 52, 24, ORANGE);
    }
}

void SimulationRenderer::DrawHudPanel(const Simulation& simulation) const
{
    const int hudX = config_.screenWidth - config_.hudWidth;
    const Rectangle hudBounds{
        static_cast<float>(hudX),
        0.0f,
        static_cast<float>(config_.hudWidth),
        static_cast<float>(config_.screenHeight)
    };

    DrawRectangleRec(hudBounds, {28, 30, 38, 255});
    DrawLine(hudX, 0, hudX, config_.screenHeight, kGridColor);

    const auto& player = simulation.GetPlayer();
    const auto& policeManager = simulation.GetPoliceManager();
    const sim::entities::PoliceNpc* primaryPolice = simulation.GetPrimaryPoliceNpc();
    const auto& timer = simulation.GetTimer();
    const auto action = primaryPolice != nullptr ? primaryPolice->GetLastAction() : sim::entities::NpcAction::None;
    const float reward = primaryPolice != nullptr ? primaryPolice->CalculateReward(player, action) : 0.0f;
    const float distance = primaryPolice != nullptr
                               ? sim::math::Distance(player.GetPosition(), primaryPolice->GetPosition())
                               : 0.0f;

    int y = 20;
    const int x = hudX + 16;
    const int lineHeight = 22;

    auto drawLine = [&](const char* text, Color color) {
        DrawText(text, x, y, 18, color);
        y += lineHeight;
    };

    auto drawSection = [&](const char* title) {
        DrawText(title, x, y, 20, kAccent);
        y += 28;
    };

    drawSection("Simulation");
    char buffer[256];

    std::snprintf(buffer, sizeof(buffer), "Tick: %llu", static_cast<unsigned long long>(timer.GetTickIndex()));
    drawLine(buffer, kTextPrimary);

    std::snprintf(buffer, sizeof(buffer), "Time: %.1fs  dt: %.2fs", timer.GetElapsedTime(), timer.GetDeltaTime());
    drawLine(buffer, kTextPrimary);

    std::snprintf(buffer, sizeof(buffer), "Rate: %.0f Hz", timer.GetTickRateHz());
    drawLine(buffer, kTextMuted);

    y += 8;
    drawSection("Player");
    std::snprintf(buffer, sizeof(buffer), "Pos: (%.1f, %.1f)", player.GetPosition().x, player.GetPosition().y);
    drawLine(buffer, kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "HP: %.0f / %.0f", player.GetHealth(), sim::entities::Player::kMaxHealth);
    drawLine(buffer, player.GetHealth() < 40.0f ? RED : kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Status: %s", sim::entities::ToString(player.GetLifeState()));
    drawLine(buffer, player.IsDead() ? RED : kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Wanted: %d / %d", player.GetWantedLevel(), sim::entities::Player::kMaxWantedLevel);
    drawLine(buffer, player.GetWantedLevel() > 0 ? ORANGE : kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Pursuit: %.1f / %.1fs",
                  simulation.GetPursuitTimerSeconds(), simulation.GetPursuitFailureSeconds());
    drawLine(buffer, simulation.GetPursuitTimerSeconds() > 7.0f ? ORANGE : kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Speed: %.1f u/s", simulation.GetPlayerCurrentSpeed());
    drawLine(buffer, simulation.GetPlayerCurrentSpeed() > 10.0f ? ORANGE : kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Heading: (%.2f, %.2f)",
                  simulation.GetPlayerCurrentHeading().x, simulation.GetPlayerCurrentHeading().y);
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Desired: (%.2f, %.2f)",
                  simulation.GetPlayerDesiredHeading().x, simulation.GetPlayerDesiredHeading().y);
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Steer: (%.2f, %.2f)",
                  simulation.GetPlayerSteeringVector().x, simulation.GetPlayerSteeringVector().y);
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Escape: (%.2f, %.2f)",
                  simulation.GetPlayerEscapeVector().x, simulation.GetPlayerEscapeVector().y);
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Perceived: %zu", simulation.GetPlayerPerceivedObstacleCount());
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Escape Score: %.2f", simulation.GetPlayerEscapeScore());
    drawLine(buffer, kTextMuted);
    std::snprintf(buffer, sizeof(buffer), "Danger: %.2f", simulation.GetPlayerDangerLevel());
    drawLine(buffer, simulation.GetPlayerDangerLevel() > 1.0f ? ORANGE : kTextMuted);

    y += 8;
    drawSection("Police");
    std::snprintf(buffer, sizeof(buffer), "Active: %zu", policeManager.GetActiveCount());
    drawLine(buffer, kTextPrimary);
    std::snprintf(buffer, sizeof(buffer), "Wanted: %d / %d", player.GetWantedLevel(), sim::entities::Player::kMaxWantedLevel);
    drawLine(buffer, player.GetWantedLevel() > 0 ? ORANGE : kTextPrimary);
    if (primaryPolice != nullptr) {
        std::snprintf(buffer, sizeof(buffer), "Lead Pos: (%.1f, %.1f)",
                      primaryPolice->GetPosition().x, primaryPolice->GetPosition().y);
        drawLine(buffer, kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Lead HP: %.0f / %.0f",
                      primaryPolice->GetHealth(), sim::entities::PoliceNpc::kMaxHealth);
        drawLine(buffer, kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "State: %s", sim::entities::ToString(primaryPolice->GetState()));
        drawLine(buffer, kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Speed x%.2f", primaryPolice->GetSpeedMultiplier());
        drawLine(buffer, primaryPolice->GetSpeedMultiplier() > 1.0f ? ORANGE : kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Effective Speed: %.1f u/s", primaryPolice->GetEffectiveMaxSpeed());
        drawLine(buffer, primaryPolice->GetCurrentSpeed() > 20.0f ? ORANGE : kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Action: %s", sim::entities::ToString(action));
        drawLine(buffer, kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Distance: %.1f", distance);
        drawLine(buffer, distance <= 10.0f ? RED : kTextPrimary);
        std::snprintf(buffer, sizeof(buffer), "Lead Heading: (%.2f, %.2f)",
                      primaryPolice->GetCurrentHeading().x, primaryPolice->GetCurrentHeading().y);
        drawLine(buffer, kTextMuted);
        std::snprintf(buffer, sizeof(buffer), "Lead Desired: (%.2f, %.2f)",
                      primaryPolice->GetDesiredHeading().x, primaryPolice->GetDesiredHeading().y);
        drawLine(buffer, kTextMuted);
        std::snprintf(buffer, sizeof(buffer), "Lead Steer: (%.2f, %.2f)",
                      primaryPolice->GetSteeringVector().x, primaryPolice->GetSteeringVector().y);
        drawLine(buffer, kTextMuted);
        std::snprintf(buffer, sizeof(buffer), "Lead Pursuit: (%.2f, %.2f)",
                      primaryPolice->GetPursuitVector().x, primaryPolice->GetPursuitVector().y);
        drawLine(buffer, kTextMuted);
    } else {
        drawLine("Lead: none", kTextMuted);
    }

    std::size_t policeIndex = 0;
    for (const auto& police : policeManager.GetPoliceNpcs()) {
        const auto& target = police.GetCurrentTarget();
        std::snprintf(buffer, sizeof(buffer), "P%zu %s Target: (%.1f, %.1f)",
                      policeIndex + 1, policeManager.GetRoleName(policeIndex), target.x, target.y);
        drawLine(buffer, kTextMuted);
        std::snprintf(buffer, sizeof(buffer), "P%zu Speed: %.1f", policeIndex + 1, police.GetCurrentSpeed());
        drawLine(buffer, police.GetCurrentSpeed() > sim::entities::PoliceNpc::kBaseSpeed ? ORANGE : kTextMuted);
        ++policeIndex;
    }

    y += 8;
    drawSection("RL Signals");
    std::snprintf(buffer, sizeof(buffer), "Reward: %.2f", reward);
    drawLine(buffer, kAccent);

    static const char* kFeatureNames[] = {
        "NPC X", "NPC Y", "Player X", "Player Y", "Distance",
        "Wanted", "Police HP", "State"
    };

    if (primaryPolice != nullptr) {
        const auto observation = primaryPolice->GetObservation(player);
        for (std::size_t index = 0; index < observation.features.size() && index < 8; ++index) {
            std::snprintf(buffer, sizeof(buffer), "%s: %.2f", kFeatureNames[index], observation.features[index]);
            drawLine(buffer, kTextMuted);
        }
    }

    if (!simulation.GetLastEventMessage().empty()) {
        y += 8;
        drawSection("Event");
        DrawText(simulation.GetLastEventMessage().c_str(), x, y, 16, ORANGE);
    }

    if (simulation.IsFinished()) {
        DrawText("Simulation complete", x, config_.screenHeight - 36, 18, kAccent);
    }
}

} // namespace sim::visualization
