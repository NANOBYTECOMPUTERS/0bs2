#include "SmartBlender.h"

#include <algorithm>
#include <cmath>

namespace aim
{
void SmartBlender::setSettings(const SmartBlenderSettings& nextSettings)
{
    settings = nextSettings;
    settings.aggression = clampFinite(settings.aggression, 0.30, 1.0, 0.65);
    settings.nearTargetDamping = clampFinite(settings.nearTargetDamping, 0.0, 1.0, 0.75);
    settings.deadzonePx = clampFinite(settings.deadzonePx, 0.0, 12.0, 0.0);
    settings.jerkLimitPx = clampFinite(settings.jerkLimitPx, 0.02, 8.0, 0.65);
    settings.confidenceFloor = clampFinite(settings.confidenceFloor, 0.0, 1.0, 0.45);
    settings.speedReferencePxPerSec = clampFinite(settings.speedReferencePxPerSec, 50.0, 5000.0, 1100.0);
}

void SmartBlender::reset()
{
    previousX = 0.0;
    previousY = 0.0;
    previousDeltaX = 0.0;
    previousDeltaY = 0.0;
    jitterScore = 0.0;
    hasPrevious = false;
}

SmartBlendOutput SmartBlender::apply(const SmartBlendInput& input)
{
    SmartBlendOutput output;
    output.x = input.desiredX;
    output.y = input.desiredY;

    const bool saneInput =
        std::isfinite(input.desiredX) &&
        std::isfinite(input.desiredY) &&
        std::isfinite(input.distance) &&
        std::isfinite(input.precisionRadius) &&
        std::isfinite(input.targetSize) &&
        std::isfinite(input.targetSpeed) &&
        std::isfinite(input.confidence) &&
        std::isfinite(input.dtSeconds);

    // Fail closed to the unmodified PID candidate whenever the advisory shaper cannot reason safely.
    if (!settings.enabled || !saneInput || input.confidence < settings.confidenceFloor)
    {
        previousX = output.x;
        previousY = output.y;
        previousDeltaX = 0.0;
        previousDeltaY = 0.0;
        jitterScore *= 0.90;
        output.jitterScore = jitterScore;
        hasPrevious = true;
        return output;
    }

    output.active = true;
    const double precisionRadius = std::max(0.0, input.precisionRadius);
    if (settings.deadzonePx > 0.0 && input.distance <= precisionRadius + settings.deadzonePx)
    {
        previousX = 0.0;
        previousY = 0.0;
        hasPrevious = true;
        output.x = 0.0;
        output.y = 0.0;
        output.alpha = 0.0;
        output.nearAmount = 1.0;
        output.jerkLimitPx = settings.jerkLimitPx;
        output.jitterScore = jitterScore;
        return output;
    }

    const double targetSize = std::max(1.0, input.targetSize);
    const double slowdownRadius = std::max(precisionRadius + 0.25, targetSize * 0.85);
    const double nearT = (input.distance - precisionRadius) / std::max(0.001, slowdownRadius - precisionRadius);
    const double nearAmount = 1.0 - smoothStep(nearT);
    const double speedFactor = smoothStep(input.targetSpeed / std::max(1.0, settings.speedReferencePxPerSec));
    const double confidenceWeight = settings.confidenceFloor >= 1.0
        ? 1.0
        : smoothStep((input.confidence - settings.confidenceFloor) / std::max(1e-6, 1.0 - settings.confidenceFloor));

    const double nearDamping = std::clamp(settings.nearTargetDamping * nearAmount * confidenceWeight, 0.0, 1.0);
    double alpha = settings.aggression * (1.0 - 0.86 * nearDamping);
    alpha += (1.0 - alpha) * 0.24 * speedFactor * (1.0 - 0.55 * nearAmount);
    alpha = std::clamp(alpha, 0.04, 1.0);

    double dampingScale = 1.0 - nearDamping * (0.48 + 0.32 * (1.0 - speedFactor));
    dampingScale = std::clamp(dampingScale, 0.12, 1.0);
    double candidateX = input.desiredX * dampingScale;
    double candidateY = input.desiredY * dampingScale;
    updateJitterScore(candidateX, candidateY, input);

    const double advisoryMagnitude =
        std::hypot(input.feedForwardX + input.learnedFeedForwardX + input.neuralRefinementX,
            input.feedForwardY + input.learnedFeedForwardY + input.neuralRefinementY);
    const double advisoryBlend = smoothStep(advisoryMagnitude / std::max(0.25, settings.jerkLimitPx * 4.0));
    const double oscillationPenalty = std::clamp(jitterScore * (0.35 + 0.65 * advisoryBlend), 0.0, 0.75);
    alpha *= 1.0 - 0.55 * oscillationPenalty;
    dampingScale *= 1.0 - 0.45 * oscillationPenalty;
    candidateX = input.desiredX * dampingScale;
    candidateY = input.desiredY * dampingScale;

    if (!hasPrevious)
    {
        previousX = candidateX;
        previousY = candidateY;
        hasPrevious = true;
    }

    double blendedX = previousX + (candidateX - previousX) * alpha;
    double blendedY = previousY + (candidateY - previousY) * alpha;

    double jerkLimit = settings.jerkLimitPx;
    jerkLimit *= 0.55 + 0.45 * speedFactor;
    jerkLimit *= 1.0 - 0.30 * nearAmount;
    jerkLimit *= 1.0 - 0.35 * oscillationPenalty;
    jerkLimit = std::max(0.01, jerkLimit);

    const double changeX = blendedX - previousX;
    const double changeY = blendedY - previousY;
    const double changeMag = std::hypot(changeX, changeY);
    if (changeMag > jerkLimit && changeMag > 1e-9)
    {
        const double scale = jerkLimit / changeMag;
        blendedX = previousX + changeX * scale;
        blendedY = previousY + changeY * scale;
    }

    const double candidateMag = std::hypot(candidateX, candidateY);
    const double blendedMag = std::hypot(blendedX, blendedY);
    if (candidateMag > 1e-9 && blendedMag > 1e-9)
    {
        const double directionDot = candidateX * blendedX + candidateY * blendedY;
        if (directionDot < 0.0)
        {
            blendedX = candidateX * std::min(0.50, alpha);
            blendedY = candidateY * std::min(0.50, alpha);
        }
    }

    if (!std::isfinite(blendedX) || !std::isfinite(blendedY))
    {
        // Fail closed to the unmodified PID candidate on numeric instability.
        blendedX = input.desiredX;
        blendedY = input.desiredY;
        output.active = false;
    }

    previousX = blendedX;
    previousY = blendedY;
    output.x = blendedX;
    output.y = blendedY;
    output.alpha = alpha;
    output.jerkLimitPx = jerkLimit;
    output.nearAmount = nearAmount;
    output.jitterScore = jitterScore;
    output.oscillationPenalty = oscillationPenalty;
    return output;
}

void SmartBlender::updateJitterScore(double nextX, double nextY, const SmartBlendInput& input)
{
    if (!hasPrevious)
    {
        previousDeltaX = 0.0;
        previousDeltaY = 0.0;
        jitterScore *= 0.90;
        return;
    }

    const double deltaX = nextX - previousX;
    const double deltaY = nextY - previousY;
    const double deltaMag = std::hypot(deltaX, deltaY);
    const double previousDeltaMag = std::hypot(previousDeltaX, previousDeltaY);
    const double reversalDot = deltaX * previousDeltaX + deltaY * previousDeltaY;

    jitterScore *= 0.90;
    if (deltaMag > 0.01 && previousDeltaMag > 0.01 && reversalDot < 0.0)
        jitterScore += std::clamp(deltaMag / std::max(0.25, settings.jerkLimitPx * 3.0), 0.0, 0.40);

    const double learnedMag = std::hypot(input.learnedFeedForwardX, input.learnedFeedForwardY);
    const double neuralMag = std::hypot(input.neuralRefinementX, input.neuralRefinementY);
    if (input.distance < input.targetSize * 0.75 && (learnedMag + neuralMag) > settings.jerkLimitPx)
        jitterScore += 0.08;

    jitterScore = std::clamp(jitterScore, 0.0, 1.0);
    previousDeltaX = deltaX;
    previousDeltaY = deltaY;
}

double SmartBlender::clampFinite(double value, double lo, double hi, double fallback)
{
    if (!std::isfinite(value))
        return fallback;
    return std::clamp(value, lo, hi);
}

double SmartBlender::smoothStep(double value)
{
    value = std::clamp(value, 0.0, 1.0);
    return value * value * (3.0 - 2.0 * value);
}
}
