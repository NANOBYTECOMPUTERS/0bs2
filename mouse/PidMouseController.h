#ifndef PID_MOUSE_CONTROLLER_H
#define PID_MOUSE_CONTROLLER_H

#include "PidGovernor.h"

#include <chrono>
#include <memory>
#include <utility>

namespace aim
{
struct PidMouseSettings
{
    int actuatorHz = 1000;
    double kp = 0.0085;
    double ki = 0.0003;
    double kd = 0.0001;
    double deadzonePx = 0.0;
    double maxPixelStep = 0.80;
    double outputScale = 0.10;
    double minOutputScale = 0.02;
    double maxOutputScale = 0.35;
    double sizeReferencePx = 48.0;
    double sizeMinScale = 0.20;
    double sizeMaxScale = 1.00;
    double precisionRadiusScale = 0.012;
    double slowdownRadiusScale = 0.30;
    double overshootBrake = 0.35;
    double divergenceBoost = 0.35;
    double scaleResponse = 8.0;
    double maxIntegral = 120.0;
    double maxDerivativeTerm = 0.02;
    double derivativeFilterTauMs = 18.0;
    double targetLossTimeoutMs = 90.0;
    bool feedForwardEnabled = true;
    double feedForwardGain = 0.35;
    double feedForwardLookaheadMs = 24.0;
    double feedForwardMaxStep = 0.35;
    double feedForwardMinSpeed = 20.0;
    double feedForwardConfidenceFloor = 0.55;
    bool governorEnabled = false;
    double governorBlend = 1.0;
    double governorMaxSpeedMultiple = 5.0;
};

struct PidMouseObservation
{
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double targetOffsetX = 0.0;
    double targetOffsetY = 0.0;
    double confidence = 1.0;
    std::chrono::steady_clock::time_point timestamp{};
    bool valid = false;
};

struct PidMouseCommand
{
    double pixelDx = 0.0;
    double pixelDy = 0.0;
    double errorX = 0.0;
    double errorY = 0.0;
    double pX = 0.0;
    double pY = 0.0;
    double iX = 0.0;
    double iY = 0.0;
    double dX = 0.0;
    double dY = 0.0;
    double outputScale = 0.0;
    double targetSize = 0.0;
    double governorKpScale = 1.0;
    double governorKiScale = 1.0;
    double governorKdScale = 1.0;
    double governorSpeedScale = 1.0;
    double feedForwardX = 0.0;
    double feedForwardY = 0.0;
    double feedForwardScale = 0.0;
    bool governorActive = false;
    bool feedForwardActive = false;
    bool active = false;
};

class PidMouseController
{
public:
    void setSettings(const PidMouseSettings& nextSettings);
    const PidMouseSettings& getSettings() const { return settings; }
    void setGovernor(std::shared_ptr<IPidGovernor> nextGovernor);
    void reset();
    void updateObservation(const PidMouseObservation& observation, double centerX, double centerY);
    PidMouseCommand step(std::chrono::steady_clock::time_point now);

private:
    struct AxisState
    {
        double integral = 0.0;
        double filteredDerivative = 0.0;
        double previousError = 0.0;
        bool hasPreviousError = false;
    };

    PidMouseSettings settings;
    PidMouseObservation latest;
    bool hasObservation = false;
    std::chrono::steady_clock::time_point lastStep{};
    AxisState axisX;
    AxisState axisY;
    double virtualErrorX = 0.0;
    double virtualErrorY = 0.0;
    double previousDistance = 0.0;
    double outputScaleState = 0.10;
    double targetVx = 0.0;
    double targetVy = 0.0;
    double targetAx = 0.0;
    double targetAy = 0.0;
    double previousOutputX = 0.0;
    double previousOutputY = 0.0;
    double movementSinceObservationX = 0.0;
    double movementSinceObservationY = 0.0;
    double feedForwardCooldownSeconds = 0.0;
    bool hasPreviousDistance = false;
    std::shared_ptr<IPidGovernor> governor;

    static double clampFinite(double value, double lo, double hi, double fallback);
    static double smoothAlpha(double dtSeconds, double tauSeconds);
    static double smoothStep(double t);
    double targetSize() const;
    double computeOutputScale(double distance, double dtSeconds, bool overshot);
    std::pair<double, double> computeFeedForward(
        const PidMouseCommand& command,
        double distance,
        double closingRate,
        bool overshot,
        double dtSeconds,
        double precisionRadius,
        double& feedForwardScale) const;
    PidGovernorInput buildGovernorInput(const PidMouseCommand& command, double distance, double closingRate, double overshootRisk, double dtSeconds) const;
    double computeAxis(AxisState& axis, double error, double kp, double ki, double kd, double dtSeconds, double& pTerm, double& iTerm, double& dTerm) const;
};
}

#endif
