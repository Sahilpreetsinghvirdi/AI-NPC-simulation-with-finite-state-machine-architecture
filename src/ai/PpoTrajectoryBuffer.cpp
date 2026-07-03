#include "sim/ai/PpoTrajectoryBuffer.hpp"

#include <utility>

namespace sim::ai {

PpoTrajectoryBuffer::PpoTrajectoryBuffer(const std::size_t capacity)
    : capacity_(capacity)
{
}

void PpoTrajectoryBuffer::Add(PpoTrajectoryStep step)
{
    if (capacity_ == 0) {
        return;
    }

    if (steps_.size() >= capacity_) {
        steps_.erase(steps_.begin());
    }
    steps_.push_back(std::move(step));
}

void PpoTrajectoryBuffer::Clear()
{
    steps_.clear();
}

bool PpoTrajectoryBuffer::IsReady() const
{
    return capacity_ > 0 && steps_.size() >= capacity_;
}

std::size_t PpoTrajectoryBuffer::Size() const
{
    return steps_.size();
}

std::size_t PpoTrajectoryBuffer::Capacity() const
{
    return capacity_;
}

const std::vector<PpoTrajectoryStep>& PpoTrajectoryBuffer::Steps() const
{
    return steps_;
}

} // namespace sim::ai
