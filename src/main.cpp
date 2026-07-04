#include "sim/Simulation.hpp"
#include "sim/ai/AiDebugSnapshot.hpp"
#include "sim/core/Logger.hpp"
#include "sim/visualization/AiDebugWindows.hpp"
#include "sim/visualization/SimulationRenderer.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

std::filesystem::path GetSnapshotPath()
{
    return std::filesystem::path(SIM_LOGS_DIR) / "live_ai_debug_snapshot.txt";
}

#if defined(_WIN32)
void LaunchCompanionWindow(const char* mode, const std::filesystem::path& snapshotPath)
{
    char executablePath[MAX_PATH]{};
    GetModuleFileNameA(nullptr, executablePath, MAX_PATH);

    std::string command = "\"";
    command += executablePath;
    command += "\" ";
    command += mode;
    command += " \"";
    command += snapshotPath.string();
    command += "\"";

    STARTUPINFOA startupInfo{};
    startupInfo.cb = sizeof(startupInfo);
    PROCESS_INFORMATION processInfo{};
    if (CreateProcessA(nullptr,
                       command.data(),
                       nullptr,
                       nullptr,
                       FALSE,
                       0,
                       nullptr,
                       nullptr,
                       &startupInfo,
                       &processInfo)) {
        CloseHandle(processInfo.hThread);
        CloseHandle(processInfo.hProcess);
    }
}
#endif

} // namespace

int main(const int argc, char** argv)
{
    if (argc >= 3 && std::string(argv[1]) == "--nn-window") {
        return sim::visualization::RunNeuralNetworkWindow(argv[2]);
    }

    if (argc >= 3 && std::string(argv[1]) == "--ppo-window") {
        return sim::visualization::RunPpoDashboardWindow(argv[2]);
    }

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
    const std::filesystem::path snapshotPath = GetSnapshotPath();

#if defined(_WIN32)
    LaunchCompanionWindow("--nn-window", snapshotPath);
    LaunchCompanionWindow("--ppo-window", snapshotPath);
#endif

    while (!renderer.ShouldClose()) {
        renderer.UpdateSimulation(simulation, renderer.GetFrameDeltaSeconds());
        renderer.Draw(simulation);
        sim::ai::WriteAiDebugSnapshot(simulation.BuildAiDebugSnapshot(1.0f / std::max(renderer.GetFrameDeltaSeconds(), 0.0001f)),
                                      snapshotPath);
    }

    logger.Info("Simulation closed at t=" + std::to_string(simulation.GetTimer().GetElapsedTime()) + "s");
    return 0;
}
