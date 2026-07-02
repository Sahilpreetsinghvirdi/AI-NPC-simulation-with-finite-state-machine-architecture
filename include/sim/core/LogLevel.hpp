#pragma once

namespace sim::core {

// Severity ordering matches numeric values so level filtering is a simple comparison.
enum class LogLevel {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Fatal = 5
};

const char* ToString(LogLevel level);

} // namespace sim::core
