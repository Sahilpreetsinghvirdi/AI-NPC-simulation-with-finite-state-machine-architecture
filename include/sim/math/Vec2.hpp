#pragma once

#include <cmath>
#include <ostream>
#include <string>

namespace sim::math {

// Lightweight 2D vector for world positions, offsets, and observation features.
// Uses float for game-scale coordinates and future ML tensor compatibility.
struct Vec2 {
    float x{0.0f};
    float y{0.0f};

    constexpr Vec2() = default;
    constexpr Vec2(const float xValue, const float yValue) : x(xValue), y(yValue) {}

    constexpr Vec2 operator+(const Vec2& other) const { return {x + other.x, y + other.y}; }
    constexpr Vec2 operator-(const Vec2& other) const { return {x - other.x, y - other.y}; }
    constexpr Vec2 operator*(const float scalar) const { return {x * scalar, y * scalar}; }
    constexpr Vec2 operator/(const float scalar) const { return {x / scalar, y / scalar}; }

    Vec2& operator+=(const Vec2& other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vec2& operator-=(const Vec2& other)
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    [[nodiscard]] float LengthSquared() const { return (x * x) + (y * y); }
    [[nodiscard]] float Length() const { return std::sqrt(LengthSquared()); }

    // Returns a unit vector; zero-length input yields zero to avoid NaNs in AI pipelines.
    [[nodiscard]] Vec2 Normalized() const
    {
        const float length = Length();
        if (length <= kEpsilon) {
            return {};
        }
        return {x / length, y / length};
    }

    [[nodiscard]] float Dot(const Vec2& other) const { return (x * other.x) + (y * other.y); }

    [[nodiscard]] std::string ToString() const;

    static constexpr float kEpsilon = 1e-5f;
};

inline Vec2 operator*(const float scalar, const Vec2& vector)
{
    return vector * scalar;
}

[[nodiscard]] inline float DistanceSquared(const Vec2& a, const Vec2& b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return (dx * dx) + (dy * dy);
}

[[nodiscard]] inline float Distance(const Vec2& a, const Vec2& b)
{
    return std::sqrt(DistanceSquared(a, b));
}

[[nodiscard]] inline bool ApproxEqual(const Vec2& a, const Vec2& b, const float epsilon = Vec2::kEpsilon)
{
    return std::fabs(a.x - b.x) <= epsilon && std::fabs(a.y - b.y) <= epsilon;
}

inline std::ostream& operator<<(std::ostream& stream, const Vec2& vector)
{
    return stream << '(' << vector.x << ", " << vector.y << ')';
}

} // namespace sim::math
