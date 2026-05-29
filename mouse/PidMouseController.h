#ifndef PID_MOUSE_CONTROLLER_H
#define PID_MOUSE_CONTROLLER_H

#include "PidGovernor.h"
#include "SmartBlender.h"

#include <chrono>
#include <memory>
#include <utility>

namespace aim
{
struct PidMouseSettings
{
    // Runtime sweep gate for experimental PID behavior; false keeps the legacy control path.
    bool runtimeLatencySweepEnabled = false;
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
    // Extra 0-2 frame velocity lead added to feed-forward while runtimeLatencySweepEnabled is true.
    int feedForwardFrameLookahead = 1;
    double feedForwardMaxStep = 0.35;
    double feedForwardMinSpeed = 20.0;
    double feedForwardConfidenceFloor = 0.55;
    // Uses perspective FOV geometry in center-equivalent angular units when runtimeLatencySweepEnabled is true.
    bool perspectiveFovMappingEnabled = false;
    double projectionWidthPx = 0.0;
    double projectionHeightPx = 0.0;
    double fovXDeg = 90.0;
    double fovYDeg = 60.0;
    // Integrates only below this per-axis error and when the last actuator step was not saturated.
    bool conditionalIntegrationEnabled = true;
    double conditionalIntegrationErrorPx = 12.0;
    // Error-magnitude scaling reference used while runtimeLatencySweepEnabled is true.
    bool adaptiveOutputScalingEnabled = true;
    double adaptiveOutputErrorScale = 96.0;
    // Multiplies derivative tau, range 1.0-6.0, to suppress noise-induced corrections.
    double derivativeSmoothingMultiplier = 1.5;
    bool governorEnabled = false;
    double governorBlend = 1.0;
    double governorMaxSpeedMultiple = 5.0;
    bool pid_smart_blending_enabled = false;
    double pid_smart_blending_aggression = 0.65;
    double pid_smart_blending_near_damping = 0.75;
    double pid_smart_blending_deadzone_px = 0.0;
    double pid_smart_blending_jerk_limit_px = 0.65;
    double pid_smart_blending_confidence_floor = 0.45;
};

struct PidMouseObservation
{
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
    double targetOffsetX = 0.0;
    double targetOffsetY = 0.0;
    double learnedPredictionLeadX = 0.0;
    double learnedPredictionLeadY = 0.0;
    double confidence = 1.0;
    std::chrono::steady_clock::time_point timestamp{};
    bool learnedPredictionLeadValid = false;
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
    double learnedFeedForwardX = 0.0;
    double learnedFeedForwardY = 0.0;
    double feedForwardScale = 0.0;
    double angularDxDeg = 0.0;
    double angularDyDeg = 0.0;
    double smartBlendAlpha = 1.0;
    double smartBlendJerkLimitPx = 0.0;
    double smartBlendNearAmount = 0.0;
    double smartBlendJitterScore = 0.0;
    double smartBlendOscillationPenalty = 0.0;
    bool governorActive = false;
    bool feedForwardActive = false;
    bool smartBlendActive = false;
    bool angularOutputActive = false;
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
    double observationTravelBudget = 0.0;
    double observationTravelUsed = 0.0;
    double latestControlErrorX = 0.0;
    double latestControlErrorY = 0.0;
    double feedForwardCooldownSeconds = 0.0;
    bool outputSaturatedLastStep = false;
    bool hasPreviousDistance = false;
    std::shared_ptr<IPidGovernor> governor;
    SmartBlender smartBlender;

    static double clampFinite(double value, double lo, double hi, double fallback);
    static double smoothAlpha(double dtSeconds, double tauSeconds);
    static double smoothStep(double t);
    static double degreesToRadians(double degrees);
    static double radiansToDegrees(double radians);
    double targetSize() const;
    bool perspectiveFovActive() const;
    double centerDegPerControlX() const;
    double centerDegPerControlY() const;
    double screenOffsetToControlX(double offsetPx) const;
    double screenOffsetToControlY(double offsetPx) const;
    double controlDeltaToDegreesX(double controlDelta) const;
    double controlDeltaToDegreesY(double controlDelta) const;
    void applyConvergenceDirectionGuard(double& outX, double& outY, double distance, double precisionRadius) const;
    double computeOutputScale(double distance, double dtSeconds, bool overshot);
    std::pair<double, double> computeFeedForward(
        PidMouseCommand& command,
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
