#include "sim/ai/Observation.hpp"

#include <sstream>

namespace sim::ai {

std::string Observation::ToString() const
{
    std::ostringstream stream;
    stream << "Observation{dim=" << features.size() << " values=[";

    for (std::size_t index = 0; index < features.size(); ++index) {
        if (index > 0) {
            stream << ", ";
        }
        stream << features[index];
    }

    stream << "]}";
    return stream.str();
}

} // namespace sim::ai
