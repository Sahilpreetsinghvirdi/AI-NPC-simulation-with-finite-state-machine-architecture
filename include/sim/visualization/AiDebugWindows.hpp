#pragma once

#include <filesystem>

namespace sim::visualization {

int RunNeuralNetworkWindow(const std::filesystem::path& snapshotPath);
int RunPpoDashboardWindow(const std::filesystem::path& snapshotPath);

} // namespace sim::visualization
