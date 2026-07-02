#pragma once

#include "sim/Simulation.hpp"
#include "sim/math/Vec2.hpp"

namespace sim::visualization {

// Real-time 2D view of the simulation using raylib.
class SimulationRenderer {
public:
    struct Config {
        int screenWidth{1280};
        int screenHeight{720};
        float worldWidth{120.0f};
        float worldHeight{120.0f};
        int hudWidth{360};
        int worldMargin{40};
    };

    SimulationRenderer();
    explicit SimulationRenderer(Config config);
    ~SimulationRenderer();

    SimulationRenderer(const SimulationRenderer&) = delete;
    SimulationRenderer& operator=(const SimulationRenderer&) = delete;

    [[nodiscard]] bool ShouldClose() const;
    [[nodiscard]] float GetFrameDeltaSeconds() const;
    void UpdateSimulation(Simulation& simulation, float frameDeltaSeconds);
    void Draw(const Simulation& simulation);

private:
    struct ScreenPoint {
        float x{0.0f};
        float y{0.0f};
    };

    [[nodiscard]] ScreenPoint WorldToScreen(sim::math::Vec2 worldPosition) const;
    void HandleWindowControls();
    void SyncWindowSize();
    void DrawWorldPanel(const Simulation& simulation);
    void DrawHudPanel(const Simulation& simulation) const;

    Config config_;
    float tickAccumulator_{0.0f};
    int windowedWidth_{1280};
    int windowedHeight_{720};
    bool fullscreen_{false};
};

} // namespace sim::visualization
