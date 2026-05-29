#include "PidMouseController.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace aim
{
namespace
{
constexpr double FeedForwardOvershootCooldownSeconds = 0.035;
// Allow one observation to converge fully so actuator speed is not tied to overlay repaint cadence.
constexpr double ObservationTravelBudgetFraction = 1.0;
constexpr double Pi = 3.14159265358979323846;
}

void PidMouseController::setSettings(const PidMouseSettings& nextSettings)
{
    settings = nextSettings;
    settings.actuatorHz = static_cast<int>(clampFinite(settings.actuatorHz, 30.0, 2000.0, 1000.0));
    settings.kp = clampFinite(settings.kp, 0.0, 5.0, 0.0085);
    settings.ki = clampFinite(settings.ki, 0.0, 2.0, 0.0003);
    settings.kd = clampFinite(settings.kd, 0.0, 1.0, 0.0001);
    settings.deadzonePx = clampFinite(settings.deadzonePx, 0.0, 10.0, 0.0);
    settings.maxPixelStep = clampFinite(settings.maxPixelStep, 0.01, 80.0, 0.80);
    settings.outputScale = clampFinite(settings.outputScale, 0.01, 3.0, 0.10);
    settings.minOutputScale = clampFinite(settings.minOutputScale, 0.0, 3.0, 0.02);
    settings.maxOutputScale = clampFinite(settings.maxOutputScale, 0.01, 3.0, 0.35);
    if (settings.minOutputScale > settings.maxOutputScale)
        std::swap(settings.minOutputScale, settings.maxOutputScale);
    settings.sizeReferencePx = clampFinite(settings.sizeReferencePx, 1.0, 640.0, 48.0);
    settings.sizeMinScale = clampFinite(settings.sizeMinScale, 0.01, 2.0, 0.20);
    settings.sizeMaxScale = clampFinite(settings.sizeMaxScale, 0.01, 2.0, 1.0);
    if (settings.sizeMinScale > settings.sizeMaxScale)
        std::swap(settings.sizeMinScale, settings.sizeMaxScale);
    settings.precisionRadiusScale = clampFinite(settings.precisionRadiusScale, 0.0, 0.25, 0.012);
    settings.slowdownRadiusScale = clampFinite(settings.slowdownRadiusScale, 0.01, 2.0, 0.30);
    settings.overshootBrake = clampFinite(settings.overshootBrake, 0.01, 1.0, 0.35);
    settings.divergenceBoost = clampFinite(settings.divergenceBoost, 0.0, 3.0, 0.35);
    settings.scaleResponse = clampFinite(settings.scaleResponse, 0.1, 60.0, 8.0);
    settings.maxIntegral = clampFinite(settings.maxIntegral, 0.0, 10000.0, 120.0);
    settings.maxDerivativeTerm = clampFinite(settings.maxDerivativeTerm, 0.0, 20.0, 0.02);
    settings.derivativeFilterTauMs = clampFinite(settings.derivativeFilterTauMs, 0.0, 250.0, 18.0);
    settings.targetLossTimeoutMs = clampFinite(settings.targetLossTimeoutMs, 10.0, 1000.0, 90.0);
    settings.feedForwardGain = clampFinite(settings.feedForwardGain, 0.0, 4.0, 0.35);
    settings.feedForwardLookaheadMs = clampFinite(settings.feedForwardLookaheadMs, 0.0, 120.0, 24.0);
    settings.feedForwardFrameLookahead = static_cast<int>(clampFinite(settings.feedForwardFrameLookahead, 0.0, 2.0, 1.0));
    settings.feedForwardMaxStep = clampFinite(settings.feedForwardMaxStep, 0.0, 5.0, 0.35);
    settings.feedForwardMinSpeed = clampFinite(settings.feedForwardMinSpeed, 0.0, 3000.0, 20.0);
    settings.feedForwardConfidenceFloor = clampFinite(settings.feedForwardConfidenceFloor, 0.0, 1.0, 0.55);
    settings.projectionWidthPx = clampFinite(settings.projectionWidthPx, 0.0, 10000.0, 0.0);
    settings.projectionHeightPx = clampFinite(settings.projectionHeightPx, 0.0, 10000.0, 0.0);
    settings.fovXDeg = clampFinite(settings.fovXDeg, 1.0, 179.0, 90.0);
    settings.fovYDeg = clampFinite(settings.fovYDeg, 1.0, 179.0, 60.0);
    settings.conditionalIntegrationErrorPx = clampFinite(settings.conditionalIntegrationErrorPx, 0.0, 240.0, 12.0);
    settings.adaptiveOutputErrorScale = clampFinite(settings.adaptiveOutputErrorScale, 1.0, 640.0, 96.0);
    settings.derivativeSmoothingMultiplier = clampFinite(settings.derivativeSmoothingMultiplier, 1.0, 6.0, 1.5);
    settings.governorBlend = clampFinite(settings.governorBlend, 0.0, 1.0, 1.0);
    settings.governorMaxSpeedMultiple = clampFinite(settings.governorMaxSpeedMultiple, 1.0, 5.0, 5.0);
    settings.pid_smart_blending_aggression = clampFinite(settings.pid_smart_blending_aggression, 0.30, 1.0, 0.65);
    settings.pid_smart_blending_near_damping = clampFinite(settings.pid_smart_blending_near_damping, 0.0, 1.0, 0.75);
    settings.pid_smart_blending_deadzone_px = clampFinite(settings.pid_smart_blending_deadzone_px, 0.0, 12.0, 0.0);
    settings.pid_smart_blending_jerk_limit_px = clampFinite(settings.pid_smart_blending_jerk_limit_px, 0.02, 8.0, 0.65);
    settings.pid_smart_blending_confidence_floor = clampFinite(settings.pid_smart_blending_confidence_floor, 0.0, 1.0, 0.45);
    outputScaleState = std::clamp(outputScaleState, settings.minOutputScale, settings.maxOutputScale);

    SmartBlenderSettings blendSettings;
    blendSettings.enabled = settings.pid_smart_blending_enabled;
    blendSettings.aggression = settings.pid_smart_blending_aggression;
    blendSettings.nearTargetDamping = settings.pid_smart_blending_near_damping;
    blendSettings.deadzonePx = settings.pid_smart_blending_deadzone_px;
    blendSettings.jerkLimitPx = settings.pid_smart_blending_jerk_limit_px;
    blendSettings.confidenceFloor = settings.pid_smart_blending_confidence_floor;
    smartBlender.setSettings(blendSettings);
}

void PidMouseController::setGovernor(std::shared_ptr<IPidGovernor> nextGovernor)
{
    governor = std::move(nextGovernor);
}

void PidMouseController::reset()
{
    latest = {};
    hasObservation = false;
    lastStep = {};
    axisX = {};
    axisY = {};
    virtualErrorX = 0.0;
    virtualErrorY = 0.0;
    previousDistance = 0.0;
    outputScaleState = settings.outputScale;
    targetVx = 0.0;
    targetVy = 0.0;
    targetAx = 0.0;
    targetAy = 0.0;
    previousOutputX = 0.0;
    previousOutputY = 0.0;
    movementSinceObservationX = 0.0;
    movementSinceObservationY = 0.0;
    observationTravelBudget = 0.0;
    observationTravelUsed = 0.0;
    latestControlErrorX = 0.0;
    latestControlErrorY = 0.0;
    feedForwardCooldownSeconds = 0.0;
    outputSaturatedLastStep = false;
    hasPreviousDistance = false;
    smartBlender.reset();
}

void PidMouseController::updateObservation(const PidMouseObservation& observation, double centerX, double centerY)
{
    if (!observation.valid || !std::isfinite(observation.x) || !std::isfinite(observation.y))
        return;

    const double rawErrorX = observation.x - centerX;
    const double rawErrorY = observation.y - centerY;
    const double nextControlErrorX = perspectiveFovActive() ? screenOffsetToControlX(rawErrorX) : rawErrorX;
    const double nextControlErrorY = perspectiveFovActive() ? screenOffsetToControlY(rawErrorY) : rawErrorY;

    if (hasObservation && latest.timestamp.time_since_epoch().count() != 0)
    {
        const double dt = std::chrono::duration<double>(observation.timestamp - latest.timestamp).count();
        if (dt > 0.0005 && dt < 1.0)
        {
            const double nextVx = (nextControlErrorX - latestControlErrorX + movementSinceObservationX) / dt;
            const double nextVy = (nextControlErrorY - latestControlErrorY + movementSinceObservationY) / dt;
            targetAx = (nextVx - targetVx) / dt;
            targetAy = (nextVy - targetVy) / dt;
            targetVx = nextVx;
            targetVy = nextVy;
        }
    }
    else
    {
        targetVx = 0.0;
        targetVy = 0.0;
        targetAx = 0.0;
        targetAy = 0.0;
    }

    latest = observation;
    latestControlErrorX = nextControlErrorX;
    latestControlErrorY = nextControlErrorY;
    hasObservation = true;
    movementSinceObservationX = 0.0;
    movementSinceObservationY = 0.0;
    virtualErrorX = nextControlErrorX;
    virtualErrorY = nextControlErrorY;

    const double observationSize = (observation.width > 0.0 && observation.height > 0.0)
        ? std::sqrt(observation.width * observation.height)
        : settings.sizeReferencePx;
    const double precisionRadius = std::max(settings.deadzonePx, observationSize * settings.precisionRadiusScale);
    const double observableTravel = std::max(0.0, std::hypot(nextControlErrorX, nextControlErrorY) - precisionRadius);
    // Spend up to the observed distance so convergence speed does not depend on UI repaint cadence.
    observationTravelBudget = observableTravel * ObservationTravelBudgetFraction;
    observationTravelUsed = 0.0;
}

PidMouseCommand PidMouseController::step(std::chrono::steady_clock::time_point now)
{
    PidMouseCommand command;
    if (!hasObservation)
        return command;

    const double targetAgeMs = std::chrono::duration<double, std::milli>(now - latest.timestamp).count();
    if (targetAgeMs > settings.targetLossTimeoutMs)
    {
        reset();
        return command;
    }

    double dtSeconds = 1.0 / static_cast<double>(settings.actuatorHz);
    if (lastStep.time_since_epoch().count() != 0)
        dtSeconds = std::chrono::duration<double>(now - lastStep).count();
    lastStep = now;
    dtSeconds = clampFinite(dtSeconds, 0.0005, 0.050, 1.0 / static_cast<double>(settings.actuatorHz));
    feedForwardCooldownSeconds = std::max(0.0, feedForwardCooldownSeconds - dtSeconds);

    command.errorX = virtualErrorX;
    command.errorY = virtualErrorY;

    const double distance = std::hypot(command.errorX, command.errorY);
    command.targetSize = targetSize();
    const double precisionRadius = std::max(settings.deadzonePx, command.targetSize * settings.precisionRadiusScale);
    if (distance <= precisionRadius)
    {
        axisX.integral *= 0.35;
        axisY.integral *= 0.35;
        axisX.filteredDerivative *= 0.35;
        axisY.filteredDerivative *= 0.35;
        previousDistance = distance;
        hasPreviousDistance = true;
        return command;
    }

    const bool hadPreviousError = axisX.hasPreviousError && axisY.hasPreviousError;
    const double oldErrorX = axisX.previousError;
    const double oldErrorY = axisY.previousError;
    const bool overshot =
        hadPreviousError &&
        (oldErrorX * command.errorX + oldErrorY * command.errorY) < 0.0;
    if (overshot)
        feedForwardCooldownSeconds = std::max(feedForwardCooldownSeconds, FeedForwardOvershootCooldownSeconds);
    const double overshootRisk = overshot ? 1.0 : 0.0;
    const double closingRate = hasPreviousDistance ? (previousDistance - distance) / std::max(dtSeconds, 1e-6) : 0.0;

    double outX = computeAxis(axisX, command.errorX, settings.kp, settings.ki, settings.kd, dtSeconds, command.pX, command.iX, command.dX);
    double outY = computeAxis(axisY, command.errorY, settings.kp, settings.ki, settings.kd, dtSeconds, command.pY, command.iY, command.dY);

    PidGovernorScales governorScales;
    if (settings.governorEnabled && governor && governor->available())
    {
        governorScales = sanitizeGovernorScales(
            governor->evaluate(buildGovernorInput(command, distance, closingRate, overshootRisk, dtSeconds)),
            settings.governorBlend);
    }

    if (governorScales.valid)
    {
        command.governorActive = true;
        command.governorKpScale = governorScales.kp;
        command.governorKiScale = governorScales.ki;
        command.governorKdScale = governorScales.kd;
        command.governorSpeedScale = governorScales.speed;
        outX = command.pX * governorScales.kp + command.iX * governorScales.ki + command.dX * governorScales.kd;
        outY = command.pY * governorScales.kp + command.iY * governorScales.ki + command.dY * governorScales.kd;
        command.outputScale = settings.outputScale;
    }
    else
    {
        command.outputScale = computeOutputScale(distance, dtSeconds, overshot);
    }

    outX *= command.outputScale;
    outY *= command.outputScale;

    double feedForwardScale = 0.0;
    const auto feedForward = computeFeedForward(
        command,
        distance,
        closingRate,
        overshot || feedForwardCooldownSeconds > 0.0,
        dtSeconds,
        precisionRadius,
        feedForwardScale);
    command.feedForwardX = feedForward.first;
    command.feedForwardY = feedForward.second;
    command.feedForwardScale = feedForwardScale;
    command.feedForwardActive = std::hypot(command.feedForwardX, command.feedForwardY) > 1e-6;
    outX += command.feedForwardX;
    outY += command.feedForwardY;
    applyConvergenceDirectionGuard(outX, outY, distance, precisionRadius);

    SmartBlendInput blendInput;
    blendInput.desiredX = outX;
    blendInput.desiredY = outY;
    blendInput.distance = distance;
    blendInput.precisionRadius = precisionRadius;
    blendInput.targetSize = command.targetSize;
    blendInput.targetSpeed = std::hypot(targetVx, targetVy);
    blendInput.confidence = std::isfinite(latest.confidence) ? latest.confidence : 0.0;
    blendInput.dtSeconds = dtSeconds;
    const SmartBlendOutput blendOutput = smartBlender.apply(blendInput);
    outX = blendOutput.x;
    outY = blendOutput.y;
    command.smartBlendActive = blendOutput.active;
    command.smartBlendAlpha = blendOutput.alpha;
    command.smartBlendJerkLimitPx = blendOutput.jerkLimitPx;
    command.smartBlendNearAmount = blendOutput.nearAmount;
    applyConvergenceDirectionGuard(outX, outY, distance, precisionRadius);

    const double outputMag = std::hypot(outX, outY);
    const double distanceLimit = std::max(0.0, distance - precisionRadius);
    const double sizeScale = command.targetSize > 0.0
        ? std::clamp(command.targetSize / settings.sizeReferencePx, settings.sizeMinScale, settings.sizeMaxScale)
        : 1.0;
    const double governorLimitScale = command.governorActive
        ? std::clamp(settings.governorMaxSpeedMultiple * command.governorSpeedScale, 0.0, settings.governorMaxSpeedMultiple)
        : 1.0;
    const double remainingObservationBudget = std::max(0.0, observationTravelBudget - observationTravelUsed);
    // Only enter the "budget exhausted" decay branch when the controller has
    // actually consumed travel. Without this guard, the first step() after a
    // reset (budget=0, used=0) would decay integrals before any observation
    // contributed, corrupting state on every target re-acquisition.
    if (remainingObservationBudget <= 1e-6 && observationTravelUsed > 1e-9)
    {
        axisX.integral *= 0.94;
        axisY.integral *= 0.94;
        axisX.filteredDerivative *= 0.90;
        axisY.filteredDerivative *= 0.90;
        outputSaturatedLastStep = true;
        previousDistance = distance;
        hasPreviousDistance = true;
        return command;
    }

    const double limit = std::min(
        std::min(settings.maxPixelStep * sizeScale * governorLimitScale, std::max(0.01, distanceLimit)),
        remainingObservationBudget);
    bool saturatedThisStep = false;
    if (outputMag > limit && outputMag > 1e-9)
    {
        saturatedThisStep = true;
        const double scale = limit / outputMag;
        outX *= scale;
        outY *= scale;
        axisX.integral *= 0.92;
        axisY.integral *= 0.92;
    }

    if (!std::isfinite(outX))
        outX = 0.0;
    if (!std::isfinite(outY))
        outY = 0.0;

    command.pixelDx = outX;
    command.pixelDy = outY;
    command.active = std::hypot(outX, outY) > 0.001;
    if (perspectiveFovActive())
    {
        command.angularDxDeg = controlDeltaToDegreesX(outX);
        command.angularDyDeg = controlDeltaToDegreesY(outY);
        command.angularOutputActive = command.active;
    }

    virtualErrorX -= outX;
    virtualErrorY -= outY;
    previousOutputX = outX;
    previousOutputY = outY;
    movementSinceObservationX += outX;
    movementSinceObservationY += outY;
    observationTravelUsed += std::hypot(outX, outY);
    outputSaturatedLastStep = saturatedThisStep;
    previousDistance = distance;
    hasPreviousDistance = true;
    return command;
}

double PidMouseController::clampFinite(double value, double lo, double hi, double fallback)
{
    if (!std::isfinite(value))
        return fallback;
    return std::clamp(value, lo, hi);
}

double PidMouseController::smoothAlpha(double dtSeconds, double tauSeconds)
{
    dtSeconds = std::max(0.0, dtSeconds);
    tauSeconds = std::max(1e-6, tauSeconds);
    return std::clamp(dtSeconds / (dtSeconds + tauSeconds), 0.0, 1.0);
}

double PidMouseController::smoothStep(double t)
{
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

double PidMouseController::degreesToRadians(double degrees)
{
    return degrees * Pi / 180.0;
}

double PidMouseController::radiansToDegrees(double radians)
{
    return radians * 180.0 / Pi;
}

double PidMouseController::targetSize() const
{
    if (latest.width > 0.0 && latest.height > 0.0)
        return std::sqrt(latest.width * latest.height);
    return settings.sizeReferencePx;
}

bool PidMouseController::perspectiveFovActive() const
{
    return
        settings.runtimeLatencySweepEnabled && settings.perspectiveFovMappingEnabled &&
        settings.projectionWidthPx > 1.0 &&
        settings.projectionHeightPx > 1.0 &&
        settings.fovXDeg > 1.0 &&
        settings.fovYDeg > 1.0;
}

double PidMouseController::centerDegPerControlX() const
{
    const double halfWidth = std::max(1.0, settings.projectionWidthPx) * 0.5;
    const double halfFov = degreesToRadians(std::clamp(settings.fovXDeg, 1.0, 179.0) * 0.5);
    return radiansToDegrees(std::tan(halfFov) / halfWidth);
}

double PidMouseController::centerDegPerControlY() const
{
    const double halfHeight = std::max(1.0, settings.projectionHeightPx) * 0.5;
    const double halfFov = degreesToRadians(std::clamp(settings.fovYDeg, 1.0, 179.0) * 0.5);
    return radiansToDegrees(std::tan(halfFov) / halfHeight);
}

double PidMouseController::screenOffsetToControlX(double offsetPx) const
{
    const double halfWidth = std::max(1.0, settings.projectionWidthPx) * 0.5;
    const double normalized = std::clamp(offsetPx / halfWidth, -4.0, 4.0);
    const double halfFov = degreesToRadians(std::clamp(settings.fovXDeg, 1.0, 179.0) * 0.5);
    const double angleDeg = radiansToDegrees(std::atan(std::tan(halfFov) * normalized));
    return angleDeg / std::max(1e-9, centerDegPerControlX());
}

double PidMouseController::screenOffsetToControlY(double offsetPx) const
{
    const double halfHeight = std::max(1.0, settings.projectionHeightPx) * 0.5;
    const double normalized = std::clamp(offsetPx / halfHeight, -4.0, 4.0);
    const double halfFov = degreesToRadians(std::clamp(settings.fovYDeg, 1.0, 179.0) * 0.5);
    const double angleDeg = radiansToDegrees(std::atan(std::tan(halfFov) * normalized));
    return angleDeg / std::max(1e-9, centerDegPerControlY());
}

double PidMouseController::controlDeltaToDegreesX(double controlDelta) const
{
    return controlDelta * centerDegPerControlX();
}

double PidMouseController::controlDeltaToDegreesY(double controlDelta) const
{
    return controlDelta * centerDegPerControlY();
}

void PidMouseController::applyConvergenceDirectionGuard(
    double& outX,
    double& outY,
    double distance,
    double precisionRadius) const
{
    if (!std::isfinite(outX) || !std::isfinite(outY) || distance <= precisionRadius + 1e-6)
        return;

    const double outputMag = std::hypot(outX, outY);
    if (outputMag <= 1e-9)
        return;

    const double radialX = virtualErrorX / std::max(distance, 1e-9);
    const double radialY = virtualErrorY / std::max(distance, 1e-9);
    double radialComponent = outX * radialX + outY * radialY;

    const double tangentX = outX - radialX * radialComponent;
    const double tangentY = outY - radialY * radialComponent;
    const double tangentMag = std::hypot(tangentX, tangentY);

    const double size = targetSize();
    const double slowdownRadius = std::max(precisionRadius + 0.25, size * settings.slowdownRadiusScale);
    const double nearT = (distance - precisionRadius) / std::max(0.001, slowdownRadius - precisionRadius);
    const double nearAmount = 1.0 - smoothStep(nearT);
    const double tangentialRatio = 0.35 - 0.17 * nearAmount;

    if (radialComponent < 0.0)
        radialComponent = 0.0;

    const double maxTangential = std::max(0.02, std::abs(radialComponent) * tangentialRatio);
    double guardedTangentX = tangentX;
    double guardedTangentY = tangentY;
    if (tangentMag > maxTangential && tangentMag > 1e-9)
    {
        const double scale = maxTangential / tangentMag;
        guardedTangentX *= scale;
        guardedTangentY *= scale;
    }

    outX = radialX * radialComponent + guardedTangentX;
    outY = radialY * radialComponent + guardedTangentY;
}

double PidMouseController::computeOutputScale(double distance, double dtSeconds, bool overshot)
{
    const double size = targetSize();
    const double precisionRadius = std::max(settings.deadzonePx, size * settings.precisionRadiusScale);
    const double slowdownRadius = std::max(precisionRadius + 0.25, size * settings.slowdownRadiusScale);
    const double nearT = (distance - precisionRadius) / std::max(0.001, slowdownRadius - precisionRadius);
    const double nearScale = smoothStep(nearT);
    const double sizeScale = std::clamp(size / settings.sizeReferencePx, settings.sizeMinScale, settings.sizeMaxScale);

    double desired = settings.outputScale;
    if (settings.runtimeLatencySweepEnabled && settings.adaptiveOutputScalingEnabled)
    {
        const double errorT = smoothStep(distance / std::max(1.0, settings.adaptiveOutputErrorScale));
        const double magnitudeScale = 0.45 + 0.55 * errorT;
        desired = settings.minOutputScale + (desired - settings.minOutputScale) * magnitudeScale;
    }

    if (hasPreviousDistance)
    {
        const double closingRate = (previousDistance - distance) / std::max(dtSeconds, 1e-6);
        if (overshot)
        {
            outputScaleState = std::max(settings.minOutputScale, outputScaleState * settings.overshootBrake);
            axisX.integral *= 0.25;
            axisY.integral *= 0.25;
            axisX.filteredDerivative *= 0.35;
            axisY.filteredDerivative *= 0.35;
        }
        else if (closingRate < -2.0 && distance > precisionRadius)
        {
            desired *= 1.0 + settings.divergenceBoost;
        }
        else if (closingRate > 2.0)
        {
            desired *= 0.80;
        }
    }

    const double alpha = std::clamp(dtSeconds * settings.scaleResponse, 0.0, 1.0);
    outputScaleState += (desired - outputScaleState) * alpha;
    outputScaleState = std::clamp(outputScaleState, settings.minOutputScale, settings.maxOutputScale);

    return std::clamp(outputScaleState * nearScale * sizeScale, 0.0, settings.maxOutputScale);
}

std::pair<double, double> PidMouseController::computeFeedForward(
    PidMouseCommand& command,
    double distance,
    double closingRate,
    bool overshot,
    double dtSeconds,
    double precisionRadius,
    double& feedForwardScale) const
{
    feedForwardScale = 0.0;
    if (overshot || settings.feedForwardMaxStep <= 0.0)
        return { 0.0, 0.0 };

    const bool learnedFeedForwardRequested = latest.learnedPredictionLeadValid;
    if (!settings.feedForwardEnabled && !learnedFeedForwardRequested)
        return { 0.0, 0.0 };

    double lookaheadSec = std::clamp(settings.feedForwardLookaheadMs * 0.001, 0.0, 0.120);
    if (settings.runtimeLatencySweepEnabled && settings.feedForwardFrameLookahead > 0)
        lookaheadSec += static_cast<double>(settings.feedForwardFrameLookahead) * std::max(dtSeconds, 1e-6);
    lookaheadSec = std::clamp(lookaheadSec, 0.0, 0.160);

    const double effectiveFeedForwardGain = settings.runtimeLatencySweepEnabled ? settings.feedForwardGain : std::min(settings.feedForwardGain, 2.0);
    if (lookaheadSec <= 1e-6)
        return { 0.0, 0.0 };

    double stepX = 0.0;
    double stepY = 0.0;
    if (settings.feedForwardEnabled && effectiveFeedForwardGain > 0.0)
    {
        const double motionSpeed = std::hypot(targetVx, targetVy);
        if (std::isfinite(motionSpeed) && motionSpeed >= settings.feedForwardMinSpeed)
        {
            const double size = targetSize();
            const double slowdownRadius = std::max(precisionRadius + 0.25, size * settings.slowdownRadiusScale);
            const double nearT = (distance - precisionRadius) / std::max(0.001, slowdownRadius - precisionRadius);
            const double nearScale = smoothStep(nearT);

            double confidenceScale = 1.0;
            const double latestConfidence = std::isfinite(latest.confidence) ? latest.confidence : 0.0;
            if (settings.feedForwardConfidenceFloor >= 1.0)
            {
                confidenceScale = latestConfidence >= 1.0 ? 1.0 : 0.0;
            }
            else
            {
                confidenceScale = std::clamp(
                    (latestConfidence - settings.feedForwardConfidenceFloor) /
                    std::max(1e-6, 1.0 - settings.feedForwardConfidenceFloor),
                    0.0,
                    1.0);
            }

            double approachScale = 1.0;
            if (closingRate > 0.0)
            {
                approachScale *= 1.0 - 0.65 * std::clamp(closingRate / std::max(1.0, motionSpeed), 0.0, 1.0);
            }

            const double motionTowardCenter = command.errorX * targetVx + command.errorY * targetVy;
            if (motionTowardCenter < 0.0 && distance < slowdownRadius * 1.5)
                approachScale *= 0.45;

            approachScale = std::clamp(approachScale, 0.0, 1.0);
            if (nearScale > 1e-6 && confidenceScale > 1e-6 && approachScale > 1e-6)
            {
                double leadX = targetVx * lookaheadSec + 0.5 * targetAx * lookaheadSec * lookaheadSec;
                double leadY = targetVy * lookaheadSec + 0.5 * targetAy * lookaheadSec * lookaheadSec;
                if (!std::isfinite(leadX)) leadX = 0.0;
                if (!std::isfinite(leadY)) leadY = 0.0;

                const double maxLead = std::max(1.0, settings.feedForwardMaxStep * (lookaheadSec / std::max(dtSeconds, 1e-6)) * 2.0);
                const double leadMag = std::hypot(leadX, leadY);
                if (leadMag > maxLead && leadMag > 1e-9)
                {
                    const double leadScale = maxLead / leadMag;
                    leadX *= leadScale;
                    leadY *= leadScale;
                }

                feedForwardScale = effectiveFeedForwardGain * nearScale * confidenceScale * approachScale;
                stepX = leadX * (dtSeconds / lookaheadSec) * feedForwardScale;
                stepY = leadY * (dtSeconds / lookaheadSec) * feedForwardScale;
            }
        }
    }

    double learnedStepX = 0.0;
    double learnedStepY = 0.0;
    if (learnedFeedForwardRequested)
    {
        const double learnedLookaheadSec = std::clamp(settings.feedForwardLookaheadMs * 0.001, 0.016, 0.160);
        learnedStepX = latest.learnedPredictionLeadX * (dtSeconds / learnedLookaheadSec);
        learnedStepY = latest.learnedPredictionLeadY * (dtSeconds / learnedLookaheadSec);

        const double learnedStepMag = std::hypot(learnedStepX, learnedStepY);
        if (learnedStepMag > settings.feedForwardMaxStep && learnedStepMag > 1e-9)
        {
            const double learnedScale = settings.feedForwardMaxStep / learnedStepMag;
            learnedStepX *= learnedScale;
            learnedStepY *= learnedScale;
        }

        if (!std::isfinite(learnedStepX)) learnedStepX = 0.0;
        if (!std::isfinite(learnedStepY)) learnedStepY = 0.0;
        command.learnedFeedForwardX = learnedStepX;
        command.learnedFeedForwardY = learnedStepY;
        stepX += learnedStepX;
        stepY += learnedStepY;
    }

    const double stepMag = std::hypot(stepX, stepY);
    if (stepMag > settings.feedForwardMaxStep && stepMag > 1e-9)
    {
        const double stepScale = settings.feedForwardMaxStep / stepMag;
        stepX *= stepScale;
        stepY *= stepScale;
    }

    if (!std::isfinite(stepX)) stepX = 0.0;
    if (!std::isfinite(stepY)) stepY = 0.0;
    return { stepX, stepY };
}

PidGovernorInput PidMouseController::buildGovernorInput(
    const PidMouseCommand& command,
    double distance,
    double closingRate,
    double overshootRisk,
    double dtSeconds) const
{
    PidGovernorInput input;
    const double offsetX = perspectiveFovActive() ? screenOffsetToControlX(latest.targetOffsetX) : latest.targetOffsetX;
    const double offsetY = perspectiveFovActive() ? screenOffsetToControlY(latest.targetOffsetY) : latest.targetOffsetY;
    input.errorX = command.errorX - offsetX;
    input.errorY = command.errorY - offsetY;
    input.errorDistance = std::hypot(input.errorX, input.errorY);
    const auto directionWeights = pidGovernorEightAxisWeights(input.errorX, input.errorY);
    input.errorDirectionRight = directionWeights[0];
    input.errorDirectionDownRight = directionWeights[1];
    input.errorDirectionDown = directionWeights[2];
    input.errorDirectionDownLeft = directionWeights[3];
    input.errorDirectionLeft = directionWeights[4];
    input.errorDirectionUpLeft = directionWeights[5];
    input.errorDirectionUp = directionWeights[6];
    input.errorDirectionUpRight = directionWeights[7];
    input.targetWidth = latest.width;
    input.targetHeight = latest.height;
    input.targetSize = command.targetSize;
    input.targetOffsetX = offsetX;
    input.targetOffsetY = offsetY;
    input.aimPointErrorX = command.errorX;
    input.aimPointErrorY = command.errorY;
    input.targetVx = targetVx;
    input.targetVy = targetVy;
    input.targetAx = targetAx;
    input.targetAy = targetAy;
    const auto motionWeights = pidGovernorMotionStateWeights(targetVx, targetVy, command.targetSize);
    input.targetMotionStill = motionWeights[0];
    input.targetMotionMoving = motionWeights[1];
    input.cursorVx = previousOutputX / std::max(dtSeconds, 1e-6);
    input.cursorVy = previousOutputY / std::max(dtSeconds, 1e-6);
    input.previousOutputX = previousOutputX;
    input.previousOutputY = previousOutputY;
    input.pidPX = command.pX;
    input.pidPY = command.pY;
    input.pidIX = command.iX;
    input.pidIY = command.iY;
    input.pidDX = command.dX;
    input.pidDY = command.dY;
    input.closingRate = closingRate;
    input.overshootRisk = overshootRisk;
    input.dt = dtSeconds;
    input.confidence = latest.confidence;
    input.maxSpeedRatio = settings.governorMaxSpeedMultiple;
    input.boxAspectRatio = latest.height > 1e-6 ? latest.width / latest.height : 1.0;
    return input;
}

double PidMouseController::computeAxis(
    AxisState& axis,
    double error,
    double kp,
    double ki,
    double kd,
    double dtSeconds,
    double& pTerm,
    double& iTerm,
    double& dTerm) const
{
    const double derivative = axis.hasPreviousError ? (error - axis.previousError) / dtSeconds : 0.0;
    const double derivativeSmoothingMultiplier =
        settings.runtimeLatencySweepEnabled ? settings.derivativeSmoothingMultiplier : 1.0;
    const double alphaD = smoothAlpha(dtSeconds, settings.derivativeFilterTauMs * 0.001 * derivativeSmoothingMultiplier);
    axis.filteredDerivative += (derivative - axis.filteredDerivative) * alphaD;

    const bool conditionalIntegrationActive =
        settings.runtimeLatencySweepEnabled && settings.conditionalIntegrationEnabled;
    const bool integrateThisStep =
        !conditionalIntegrationActive ||
        (std::abs(error) <= settings.conditionalIntegrationErrorPx && !outputSaturatedLastStep);
    if (integrateThisStep)
    {
        axis.integral += error * dtSeconds;
    }
    else
    {
        axis.integral *= 0.98;
    }
    axis.integral = std::clamp(axis.integral, -settings.maxIntegral, settings.maxIntegral);

    pTerm = kp * error;
    iTerm = ki * axis.integral;
    dTerm = std::clamp(kd * axis.filteredDerivative, -settings.maxDerivativeTerm, settings.maxDerivativeTerm);

    axis.previousError = error;
    axis.hasPreviousError = true;
    return pTerm + iTerm + dTerm;
}
}
