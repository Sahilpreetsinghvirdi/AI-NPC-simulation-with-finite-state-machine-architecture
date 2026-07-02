#include "sim/math/Vec2.hpp"

#include <sstream>

namespace sim::math {

std::string Vec2::ToString() const
{
    std::ostringstream stream;
    stream << *this;
    return stream.str();
}

} // namespace sim::math
