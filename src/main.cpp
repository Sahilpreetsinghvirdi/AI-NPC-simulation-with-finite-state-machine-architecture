#include "sim/Simulation.hpp"
#include "sim/core/Logger.hpp"
#include "sim/visualization/SimulationRenderer.hpp"

int main()
{
    sim::core::Logger::Config logConfig;
    logConfig.logToConsole = false;
    sim::core::Logger logger(logConfig);

    logger.Info("AiNpcSimulation v0.1.0 — visual sandbox online.");
    logger.Info("Log file: " + logger.GetLogFilePath().string());

    sim::Simulation simulation(logger, {
        .maxTicks = 0,
        .tickRateHz = 10.0f,
        .realTimePacing = false
    });

    sim::visualization::SimulationRenderer renderer;

    while (!renderer.ShouldClose()) {
        renderer.UpdateSimulation(simulation, renderer.GetFrameDeltaSeconds());
        renderer.Draw(simulation);
    }

    logger.Info("Simulation closed at t=" + std::to_string(simulation.GetTimer().GetElapsedTime()) + "s");
    return 0;
}
