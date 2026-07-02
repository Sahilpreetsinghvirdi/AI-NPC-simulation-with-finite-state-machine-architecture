#include "sim/core/LogLevel.hpp"

namespace sim::core {

const char* ToString(const LogLevel level)
{
    switch (level) {
    case LogLevel::Trace: return "TRACE";
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info:  return "INFO ";
    case LogLevel::Warn:  return "WARN ";
    case LogLevel::Error: return "ERROR";
    case LogLevel::Fatal: return "FATAL";
    }
    return "UNKNOWN";
}

} // namespace sim::core
