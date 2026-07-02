#pragma once

#include "sim/core/LogLevel.hpp"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>

namespace sim::core {

// Thread-safe logger with optional console and file sinks.
// Owned explicitly by simulation code (no global singleton) so tests can inject mocks.
class Logger {
public:
    struct Config {
        std::filesystem::path logDirectory;
        std::string sessionName{"simulation"};
        LogLevel minLevel{LogLevel::Info};
        bool logToConsole{true};
        bool logToFile{true};

        Config() : logDirectory(SIM_LOGS_DIR) {}
    };

    Logger();
    explicit Logger(Config config);
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void SetMinLevel(LogLevel level);
    void SetLogToConsole(bool enabled);
    void SetLogToFile(bool enabled);

    void Log(LogLevel level, std::string_view message);

    void Trace(std::string_view message) { Log(LogLevel::Trace, message); }
    void Debug(std::string_view message) { Log(LogLevel::Debug, message); }
    void Info(std::string_view message)  { Log(LogLevel::Info, message); }
    void Warn(std::string_view message)  { Log(LogLevel::Warn, message); }
    void Error(std::string_view message) { Log(LogLevel::Error, message); }
    void Fatal(std::string_view message) { Log(LogLevel::Fatal, message); }

    [[nodiscard]] const std::filesystem::path& GetLogFilePath() const { return logFilePath_; }

private:
    void WriteLineLocked(LogLevel level, std::string_view message);
    [[nodiscard]] std::string FormatTimestamp() const;
    [[nodiscard]] std::string FormatLine(LogLevel level, std::string_view message) const;
    void OpenLogFile();

    Config config_;
    std::filesystem::path logFilePath_;
    std::ofstream logFile_;
    mutable std::mutex mutex_;
};

} // namespace sim::core
