#ifndef PID_GOVERNOR_H
#define PID_GOVERNOR_H

#include <array>

namespace aim
{
constexpr int PidGovernorDirectionCount = 8;
constexpr int PidGovernorMotionStateCount = 2;
constexpr int PidGovernorFeatureCount = 40;
constexpr int PidGovernorOutputCount = 4;

struct PidGovernorInput
{
    double errorX = 0.0;
    double errorY = 0.0;
    double errorDistance = 0.0;
    double errorDirectionRight = 0.125;
    double errorDirectionDownRight = 0.125;
    double errorDirectionDown = 0.125;
    double errorDirectionDownLeft = 0.125;
    double errorDirectionLeft = 0.125;
    double errorDirectionUpLeft = 0.125;
    double errorDirectionUp = 0.125;
    double errorDirectionUpRight = 0.125;
    double targetWidth = 0.0;
    double targetHeight = 0.0;
    double targetSize = 0.0;
    double targetOffsetX = 0.0;
    double targetOffsetY = 0.0;
    double aimPointErrorX = 0.0;
    double aimPointErrorY = 0.0;
    double targetVx = 0.0;
    double targetVy = 0.0;
    double targetAx = 0.0;
    double targetAy = 0.0;
    double targetMotionStill = 1.0;
    double targetMotionMoving = 0.0;
    double cursorVx = 0.0;
    double cursorVy = 0.0;
    double previousOutputX = 0.0;
    double previousOutputY = 0.0;
    double pidPX = 0.0;
    double pidPY = 0.0;
    double pidIX = 0.0;
    double pidIY = 0.0;
    double pidDX = 0.0;
    double pidDY = 0.0;
    double closingRate = 0.0;
    double overshootRisk = 0.0;
    double dt = 0.0;
    double confidence = 1.0;
    double maxSpeedRatio = 5.0;
    double boxAspectRatio = 1.0;
};

struct PidGovernorScales
{
    double kp = 1.0;
    double ki = 1.0;
    double kd = 1.0;
    double speed = 1.0;
    bool valid = false;
};

std::array<double, PidGovernorDirectionCount> pidGovernorEightAxisWeights(double errorX, double errorY);
std::array<double, PidGovernorMotionStateCount> pidGovernorMotionStateWeights(double targetVx, double targetVy, double targetSize);
std::array<float, PidGovernorFeatureCount> pidGovernorFeatures(const PidGovernorInput& input);
PidGovernorScales sanitizeGovernorScales(const PidGovernorScales& scales, double blend);

class IPidGovernor
{
public:
    virtual ~IPidGovernor() = default;
    virtual bool available() const = 0;
    virtual PidGovernorScales evaluate(const PidGovernorInput& input) = 0;
};
}

#endif
