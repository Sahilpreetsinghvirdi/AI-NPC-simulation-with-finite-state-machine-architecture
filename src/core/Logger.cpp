#include "sim/core/Logger.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace sim::core {

namespace {

std::string FormatClockTimePoint(const std::chrono::system_clock::time_point timePoint)
{
    const auto timeT = std::chrono::system_clock::to_time_t(timePoint);
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        timePoint.time_since_epoch()) % 1000;

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &timeT);
#else
    localtime_r(&timeT, &localTime);
#endif

    std::ostringstream stream;
    stream << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << milliseconds.count();
    return stream.str();
}

std::string BuildSessionFileName(const std::string& sessionName)
{
    const auto now = std::chrono::system_clock::now();
    const auto timeT = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
#if defined(_WIN32)
    localtime_s(&localTime, &timeT);
#else
    localtime_r(&timeT, &localTime);
#endif

    std::ostringstream stream;
    stream << sessionName << '_'
           << std::put_time(&localTime, "%Y%m%d_%H%M%S") << ".log";
    return stream.str();
}

} // namespace

Logger::Logger()
    : Logger(Config{})
{
}

Logger::Logger(const Config config)
    : config_(std::move(config))
{
    if (config_.logToFile) {
        OpenLogFile();
    }
}

Logger::~Logger()
{
    std::lock_guard lock(mutex_);
    if (logFile_.is_open()) {
        logFile_.flush();
        logFile_.close();
    }
}

void Logger::SetMinLevel(const LogLevel level)
{
    std::lock_guard lock(mutex_);
    config_.minLevel = level;
}

void Logger::SetLogToConsole(const bool enabled)
{
    std::lock_guard lock(mutex_);
    config_.logToConsole = enabled;
}

void Logger::SetLogToFile(const bool enabled)
{
    std::lock_guard lock(mutex_);
    if (enabled && !config_.logToFile) {
        config_.logToFile = true;
        OpenLogFile();
        return;
    }

    config_.logToFile = enabled;
    if (!enabled && logFile_.is_open()) {
        logFile_.flush();
        logFile_.close();
    }
}

void Logger::Log(const LogLevel level, const std::string_view message)
{
    std::lock_guard lock(mutex_);
    if (level < config_.minLevel) {
        return;
    }

    WriteLineLocked(level, message);
}

void Logger::WriteLineLocked(const LogLevel level, const std::string_view message)
{
    const std::string line = FormatLine(level, message);

    if (config_.logToConsole) {
        std::ostream& stream = (level >= LogLevel::Error) ? std::cerr : std::cout;
        stream << line << '\n';
    }

    if (config_.logToFile && logFile_.is_open()) {
        logFile_ << line << '\n';
        logFile_.flush();
    }
}

std::string Logger::FormatTimestamp() const
{
    return FormatClockTimePoint(std::chrono::system_clock::now());
}

std::string Logger::FormatLine(const LogLevel level, const std::string_view message) const
{
    std::ostringstream stream;
    stream << '[' << FormatTimestamp() << "] "
           << '[' << ToString(level) << "] "
           << message;
    return stream.str();
}

void Logger::OpenLogFile()
{
    std::error_code errorCode;
    std::filesystem::create_directories(config_.logDirectory, errorCode);

    logFilePath_ = config_.logDirectory / BuildSessionFileName(config_.sessionName);
    logFile_.open(logFilePath_, std::ios::out | std::ios::app);

    if (!logFile_.is_open()) {
        config_.logToFile = false;
        std::cerr << "[Logger] Failed to open log file: " << logFilePath_.string() << '\n';
    }
}

} // namespace sim::core
