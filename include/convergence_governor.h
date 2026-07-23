#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace aim
{
struct ConvergenceGovernorSettings
{
    bool enabled = false;
    double strength = 0.65;
    double minGain = 0.20;
    double maxGain = 1.10;
};

struct ConvergenceGovernorContext
{
    int trackId = -1;
    bool observedThisFrame = true;
    int missedFrames = 0;
    int activeTrackCount = 1;
    bool kalmanEnabled = false;
    double kalmanMeasurementNoise = 0.0;
    double kalmanVelocityDamping = 0.0;
    double dtSec = 0.001;
    double stateAgeMs = 0.0;
    double distancePx = 0.0;
    double errorX = 0.0;
    double errorY = 0.0;
    double targetVelocityX = 0.0;
    double targetVelocityY = 0.0;
    double confidence = 1.0;
};

struct ConvergenceGovernorOutput
{
    bool enabled = false;
    double gainScale = 1.0;
    double maxStepScale = 1.0;
    double overshootRisk = 0.0;
    double undershootRisk = 0.0;
    double jitterRisk = 0.0;
    double lockRisk = 0.0;
    double brake = 0.0;
    double release = 0.0;
};

class ConvergenceGovernor
{
public:
    void reset()
    {
        activeTrackId_ = -1;
        hasLastError_ = false;
        hasLastMovement_ = false;
        lastErrorX_ = 0.0;
        lastErrorY_ = 0.0;
        lastDistancePx_ = 0.0;
        lastMovementX_ = 0.0;
        lastMovementY_ = 0.0;
    }

    ConvergenceGovernorOutput evaluate(
        const ConvergenceGovernorSettings& rawSettings,
        const ConvergenceGovernorContext& context)
    {
        ConvergenceGovernorSettings settings = sanitizeSettings(rawSettings);
        ConvergenceGovernorOutput out;
        if (!settings.enabled)
            return out;

        if (context.trackId != activeTrackId_)
        {
            reset();
            activeTrackId_ = context.trackId;
        }

        out.enabled = true;
        const FeatureVector features = buildFeatures(context);
        const NetworkOutput network = runNetwork(features);

        const double overshootSignal = features[11];
        const double divergingSignal = features[12];
        const double headingChangeSignal = features[13];
        const double excessiveMoveSignal = features[14];
        const double movementAgainstErrorSignal = features[15];

        out.overshootRisk = clamp01(
            0.50 * overshootSignal +
            0.25 * movementAgainstErrorSignal +
            0.15 * excessiveMoveSignal +
            0.10 * network.overshoot);
        out.undershootRisk = clamp01(
            0.36 * features[0] +
            0.26 * divergingSignal +
            0.22 * (1.0 - excessiveMoveSignal) +
            0.16 * network.undershoot);
        out.jitterRisk = clamp01(
            0.34 * headingChangeSignal +
            0.20 * features[1] +
            0.18 * features[2] +
            0.12 * features[8] +
            0.16 * network.jitter);
        out.lockRisk = clamp01(
            0.27 * features[3] +
            0.24 * features[6] +
            0.18 * features[7] +
            0.16 * features[4] +
            0.15 * network.lock);

        out.brake = clamp01(
            0.48 * out.overshootRisk +
            0.24 * out.jitterRisk +
            0.20 * out.lockRisk +
            0.08 * excessiveMoveSignal);
        out.release = clamp01(
            out.undershootRisk *
            (1.0 - 0.70 * out.overshootRisk) *
            (1.0 - 0.45 * out.jitterRisk) *
            (1.0 - 0.35 * out.lockRisk));

        const double strength = std::clamp(settings.strength, 0.0, 1.0);
        out.gainScale = std::clamp(
            1.0 + strength * (0.16 * out.release - 0.78 * out.brake),
            settings.minGain,
            settings.maxGain);
        out.maxStepScale = std::clamp(
            1.0 + strength * (0.10 * out.release - 0.58 * out.brake),
            settings.minGain,
            settings.maxGain);

        rememberError(context);
        return out;
    }

    void observeMovement(double emittedPixelX, double emittedPixelY)
    {
        if (!std::isfinite(emittedPixelX) || !std::isfinite(emittedPixelY))
            return;

        lastMovementX_ = emittedPixelX;
        lastMovementY_ = emittedPixelY;
        hasLastMovement_ = true;
    }

private:
    using FeatureVector = std::array<double, 16>;

    struct NetworkOutput
    {
        double overshoot = 0.0;
        double undershoot = 0.0;
        double jitter = 0.0;
        double lock = 0.0;
    };

    static double finiteOr(double value, double fallback)
    {
        return std::isfinite(value) ? value : fallback;
    }

    static double clamp01(double value)
    {
        return std::clamp(finiteOr(value, 0.0), 0.0, 1.0);
    }

    static double sigmoid(double value)
    {
        value = std::clamp(finiteOr(value, 0.0), -40.0, 40.0);
        return 1.0 / (1.0 + std::exp(-value));
    }

    static ConvergenceGovernorSettings sanitizeSettings(
        ConvergenceGovernorSettings settings)
    {
        settings.strength = std::clamp(finiteOr(settings.strength, 0.65), 0.0, 1.0);
        settings.minGain = std::clamp(finiteOr(settings.minGain, 0.20), 0.05, 1.0);
        settings.maxGain = std::clamp(finiteOr(settings.maxGain, 1.10), 1.0, 2.0);
        if (settings.minGain > settings.maxGain)
            settings.minGain = settings.maxGain;
        return settings;
    }

    FeatureVector buildFeatures(const ConvergenceGovernorContext& context) const
    {
        const double distancePx = std::max(0.0, finiteOr(context.distancePx, 0.0));
        const double errorX = finiteOr(context.errorX, 0.0);
        const double errorY = finiteOr(context.errorY, 0.0);
        const double speedPxPerSec = std::hypot(
            finiteOr(context.targetVelocityX, 0.0),
            finiteOr(context.targetVelocityY, 0.0));
        const double confidence = clamp01(context.confidence);
        const double stateAgeMs = std::max(0.0, finiteOr(context.stateAgeMs, 0.0));
        const double missedFrames = static_cast<double>(std::max(0, context.missedFrames));
        const double activeTracks = static_cast<double>(std::max(1, context.activeTrackCount));

        double overshoot = 0.0;
        double diverging = 0.0;
        double headingChange = 0.0;
        if (hasLastError_ && lastDistancePx_ > 0.75 && distancePx > 0.75)
        {
            const double denom = std::max(1e-6, lastDistancePx_ * distancePx);
            const double normalizedDot = std::clamp((lastErrorX_ * errorX + lastErrorY_ * errorY) / denom, -1.0, 1.0);
            overshoot = clamp01(-normalizedDot);
            diverging = clamp01((distancePx - lastDistancePx_) / std::max(12.0, lastDistancePx_));
            constexpr double pi = 3.14159265358979323846;
            headingChange = std::acos(normalizedDot) / pi;
        }

        const double movementPx = std::hypot(lastMovementX_, lastMovementY_);
        const double excessiveMove = hasLastMovement_
            ? clamp01(movementPx / std::max(1.0, distancePx))
            : 0.0;
        double movementAgainstError = 0.0;
        if (hasLastMovement_ && movementPx > 1e-6 && distancePx > 1e-6)
        {
            movementAgainstError = clamp01(
                -(lastMovementX_ * errorX + lastMovementY_ * errorY) /
                std::max(1e-6, movementPx * distancePx));
        }

        const double kalmanNoise = context.kalmanEnabled
            ? clamp01(std::log1p(std::max(0.0, finiteOr(context.kalmanMeasurementNoise, 0.0))) / std::log1p(5000.0))
            : 0.0;

        return {
            clamp01(distancePx / 160.0),
            1.0 - clamp01(distancePx / 32.0),
            clamp01(speedPxPerSec / 2400.0),
            1.0 - confidence,
            clamp01(stateAgeMs / 120.0),
            clamp01(missedFrames / 4.0),
            context.observedThisFrame ? 0.0 : clamp01((missedFrames + 1.0) / 4.0),
            clamp01((activeTracks - 1.0) / 4.0),
            kalmanNoise,
            context.kalmanEnabled ? clamp01(finiteOr(context.kalmanVelocityDamping, 0.0) / 3.0) : 0.0,
            clamp01(finiteOr(context.dtSec, 0.001) / 0.008),
            overshoot,
            diverging,
            clamp01(headingChange),
            excessiveMove,
            movementAgainstError,
        };
    }

    static NetworkOutput runNetwork(const FeatureVector& f)
    {
        const double h0 = std::tanh(-0.55 + 2.20 * f[11] + 1.15 * f[15] + 0.55 * f[14] + 0.30 * f[1]);
        const double h1 = std::tanh(-0.35 + 1.35 * f[0] + 1.10 * f[12] + 0.50 * (1.0 - f[14]));
        const double h2 = std::tanh(-0.50 + 1.70 * f[13] + 0.85 * f[2] + 0.50 * f[8] + 0.30 * f[1]);
        const double h3 = std::tanh(-0.45 + 1.40 * f[3] + 1.25 * f[6] + 0.95 * f[7] + 0.60 * f[4]);
        const double h4 = std::tanh(-0.30 + 0.95 * f[5] + 0.80 * f[10] + 0.65 * f[9]);
        const double h5 = std::tanh(0.20 + 0.80 * f[0] - 0.75 * f[1] - 0.65 * f[11] - 0.55 * f[13]);

        NetworkOutput out;
        out.overshoot = sigmoid(-1.20 + 1.75 * h0 + 0.45 * h2 + 0.30 * h4 - 0.35 * h5);
        out.undershoot = sigmoid(-1.00 + 1.55 * h1 + 0.55 * h5 - 0.70 * h0 - 0.35 * h3);
        out.jitter = sigmoid(-1.10 + 1.55 * h2 + 0.35 * h4 + 0.25 * h0);
        out.lock = sigmoid(-1.00 + 1.70 * h3 + 0.45 * h4 + 0.25 * h2);
        return out;
    }

    void rememberError(const ConvergenceGovernorContext& context)
    {
        lastErrorX_ = finiteOr(context.errorX, 0.0);
        lastErrorY_ = finiteOr(context.errorY, 0.0);
        lastDistancePx_ = std::max(0.0, finiteOr(context.distancePx, 0.0));
        hasLastError_ = true;
    }

    int activeTrackId_ = -1;
    bool hasLastError_ = false;
    bool hasLastMovement_ = false;
    double lastErrorX_ = 0.0;
    double lastErrorY_ = 0.0;
    double lastDistancePx_ = 0.0;
    double lastMovementX_ = 0.0;
    double lastMovementY_ = 0.0;
};
}
