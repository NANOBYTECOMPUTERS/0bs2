#include "PidGovernor.h"

#include <algorithm>
#include <cmath>

namespace aim
{
namespace
{
constexpr double Pi = 3.14159265358979323846;
constexpr double Tau = Pi * 2.0;
constexpr double SectorWidth = Pi / 4.0;

float feature(double value)
{
    if (!std::isfinite(value))
        return 0.0f;
    return static_cast<float>(value);
}

double sanitizeScale(double value, double fallback)
{
    if (!std::isfinite(value))
        return fallback;
    return std::clamp(value, 0.0, 1.0);
}

double smoothStep(double value)
{
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}
}

std::array<double, PidGovernorDirectionCount> pidGovernorEightAxisWeights(double errorX, double errorY)
{
    std::array<double, PidGovernorDirectionCount> weights{};
    const double distance = std::hypot(errorX, errorY);
    if (!std::isfinite(distance) || distance < 1e-6)
    {
        weights.fill(1.0 / static_cast<double>(PidGovernorDirectionCount));
        return weights;
    }

    const double angle = std::atan2(errorY, errorX);
    constexpr std::array<double, PidGovernorDirectionCount> centers{
        0.0,
        Pi / 4.0,
        Pi / 2.0,
        3.0 * Pi / 4.0,
        Pi,
        -3.0 * Pi / 4.0,
        -Pi / 2.0,
        -Pi / 4.0,
    };

    double sum = 0.0;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        const double delta = std::abs(std::remainder(angle - centers[i], Tau));
        weights[i] = std::max(0.0, 1.0 - delta / SectorWidth);
        sum += weights[i];
    }

    if (sum <= 1e-9 || !std::isfinite(sum))
    {
        weights.fill(1.0 / static_cast<double>(PidGovernorDirectionCount));
        return weights;
    }

    for (double& weight : weights)
        weight /= sum;
    return weights;
}

std::array<double, PidGovernorMotionStateCount> pidGovernorMotionStateWeights(double targetVx, double targetVy, double targetSize)
{
    const double speed = std::hypot(targetVx, targetVy);
    if (!std::isfinite(speed))
        return { 1.0, 0.0 };

    const double size = std::max(1.0, std::isfinite(targetSize) ? targetSize : 1.0);
    const double stillThreshold = std::max(0.75, size * 0.08);
    const double movingThreshold = std::max(stillThreshold + 0.75, size * 0.65);
    const double moving = smoothStep((speed - stillThreshold) / std::max(1e-6, movingThreshold - stillThreshold));
    return { 1.0 - moving, moving };
}

std::array<float, PidGovernorFeatureCount> pidGovernorFeatures(const PidGovernorInput& input)
{
    return {
        feature(input.errorX),
        feature(input.errorY),
        feature(input.errorDistance),
        feature(input.errorDirectionRight),
        feature(input.errorDirectionDownRight),
        feature(input.errorDirectionDown),
        feature(input.errorDirectionDownLeft),
        feature(input.errorDirectionLeft),
        feature(input.errorDirectionUpLeft),
        feature(input.errorDirectionUp),
        feature(input.errorDirectionUpRight),
        feature(input.targetWidth),
        feature(input.targetHeight),
        feature(input.targetSize),
        feature(input.targetOffsetX),
        feature(input.targetOffsetY),
        feature(input.aimPointErrorX),
        feature(input.aimPointErrorY),
        feature(input.targetVx),
        feature(input.targetVy),
        feature(input.targetAx),
        feature(input.targetAy),
        feature(input.targetMotionStill),
        feature(input.targetMotionMoving),
        feature(input.cursorVx),
        feature(input.cursorVy),
        feature(input.previousOutputX),
        feature(input.previousOutputY),
        feature(input.pidPX),
        feature(input.pidPY),
        feature(input.pidIX),
        feature(input.pidIY),
        feature(input.pidDX),
        feature(input.pidDY),
        feature(input.closingRate),
        feature(input.overshootRisk),
        feature(input.dt),
        feature(input.confidence),
        feature(input.maxSpeedRatio),
        feature(input.boxAspectRatio),
    };
}

PidGovernorScales sanitizeGovernorScales(const PidGovernorScales& scales, double blend)
{
    blend = std::clamp(std::isfinite(blend) ? blend : 0.0, 0.0, 1.0);

    PidGovernorScales sanitized;
    sanitized.valid = scales.valid;
    sanitized.kp = 1.0 + (sanitizeScale(scales.kp, 1.0) - 1.0) * blend;
    sanitized.ki = 1.0 + (sanitizeScale(scales.ki, 1.0) - 1.0) * blend;
    sanitized.kd = 1.0 + (sanitizeScale(scales.kd, 1.0) - 1.0) * blend;
    sanitized.speed = 1.0 + (sanitizeScale(scales.speed, 1.0) - 1.0) * blend;
    return sanitized;
}
}
