#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>

namespace aim
{
struct EgoMotionSettings
{
    bool enabled = false;
    double strength = 0.80;
    double maxShiftPx = 32.0;
    int maxAgeMs = 120;
};

struct EgoMotionShift
{
    double dx = 0.0;
    double dy = 0.0;
    bool valid = false;
};

class EgoMotionCompensator
{
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    void setSettings(const EgoMotionSettings& nextSettings)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        settings_ = sanitize(nextSettings);
        if (!settings_.enabled)
            samples_.clear();
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        samples_.clear();
    }

    void recordDelta(double dx, double dy, TimePoint timestamp)
    {
        if (!std::isfinite(dx) || !std::isfinite(dy))
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled)
            return;

        if (std::hypot(dx, dy) <= 1e-6)
            return;

        samples_.push_back({ timestamp, dx, dy });
        constexpr size_t maxSamples = 512;
        while (samples_.size() > maxSamples)
            samples_.pop_front();
    }

    EgoMotionShift consume(TimePoint start, TimePoint end, TimePoint now)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!settings_.enabled || end <= start)
        {
            samples_.clear();
            return {};
        }

        const auto maxAge = std::chrono::milliseconds(settings_.maxAgeMs);
        const TimePoint oldestAllowed = now - maxAge;

        double dx = 0.0;
        double dy = 0.0;
        bool used = false;
        std::deque<Sample> remaining;

        for (const auto& sample : samples_)
        {
            if (sample.timestamp < oldestAllowed)
                continue;

            if (sample.timestamp > end)
            {
                remaining.push_back(sample);
                continue;
            }

            if (sample.timestamp > start)
            {
                dx += sample.dx;
                dy += sample.dy;
                used = true;
            }
        }

        samples_.swap(remaining);
        if (!used)
            return {};

        dx *= settings_.strength;
        dy *= settings_.strength;
        const auto clamped = clampLength(dx, dy, settings_.maxShiftPx);
        return {
            clamped.first,
            clamped.second,
            std::hypot(clamped.first, clamped.second) > 1e-6,
        };
    }

private:
    struct Sample
    {
        TimePoint timestamp{};
        double dx = 0.0;
        double dy = 0.0;
    };

    static EgoMotionSettings sanitize(const EgoMotionSettings& in)
    {
        EgoMotionSettings out = in;
        out.strength = std::clamp(out.strength, 0.0, 1.0);
        out.maxShiftPx = std::clamp(out.maxShiftPx, 1.0, 128.0);
        out.maxAgeMs = std::clamp(out.maxAgeMs, 16, 500);
        return out;
    }

    static std::pair<double, double> clampLength(double dx, double dy, double maxLength)
    {
        const double length = std::hypot(dx, dy);
        if (!std::isfinite(length) || length <= maxLength || length <= 1e-9)
            return { dx, dy };

        const double scale = maxLength / length;
        return { dx * scale, dy * scale };
    }

    EgoMotionSettings settings_{};
    std::deque<Sample> samples_;
    std::mutex mutex_;
};
} // namespace aim
