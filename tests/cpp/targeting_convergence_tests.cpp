#include "aim_kalman.h"
#include "test_harness.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace
{
struct StepConvergenceMetrics
{
    double timeToTwoPxMs = -1.0;
    double timeToOnePxMs = -1.0;
    double finalErrorPx = 0.0;
    double pathLengthPx = 0.0;
};

StepConvergenceMetrics simulateStreamStep(double initialErrorPx, double sharpness, double maxSpeedPxPerSec)
{
    constexpr double dtSec = 0.001;
    constexpr int steps = 500;
    double applied = 0.0;
    StepConvergenceMetrics metrics;

    for (int i = 0; i < steps; ++i)
    {
        const double error = initialErrorPx - applied;
        double movement = error * std::clamp(1.0 - std::exp(-sharpness * dtSec), 0.0, 1.0);
        const double maxStep = std::max(0.01, maxSpeedPxPerSec * dtSec);
        if (std::abs(movement) > maxStep)
            movement = std::copysign(maxStep, movement);

        applied += movement;
        metrics.pathLengthPx += std::abs(movement);
        metrics.finalErrorPx = std::abs(initialErrorPx - applied);

        const double elapsedMs = static_cast<double>(i) * dtSec * 1000.0;
        if (metrics.timeToTwoPxMs < 0.0 && metrics.finalErrorPx <= 2.0)
            metrics.timeToTwoPxMs = elapsedMs;
        if (metrics.timeToOnePxMs < 0.0 && metrics.finalErrorPx <= 1.0)
            metrics.timeToOnePxMs = elapsedMs;
    }

    return metrics;
}

double deterministicNoise(int sample)
{
    return 1.2 * std::sin(static_cast<double>(sample) * 0.73) +
        0.55 * std::sin(static_cast<double>(sample) * 1.91);
}

struct EstimatorMetrics
{
    double predictionRmsPx = 0.0;
    double predictionAverageAbsPx = 0.0;
    double reversalAverageAbsPx = 0.0;
    double reversalMaxAbsPx = 0.0;
};

aim::AimKalmanSettings legacyConvergenceSettings()
{
    aim::AimKalmanSettings settings;
    settings.enabled = true;
    settings.process_noise_position = 40.0;
    settings.process_noise_velocity = 1800.0;
    settings.measurement_noise = 35.0;
    settings.velocity_damping = 0.08;
    settings.max_velocity = 20000.0;
    settings.warmup_frames = 2;
    settings.velocitySeedEnabled = true;
    settings.acquisitionFrames = 4;
    settings.runtimeLatencySweepEnabled = false;
    return settings;
}

aim::AimKalmanSettings tunedConvergenceSettings()
{
    aim::AimKalmanSettings settings = legacyConvergenceSettings();
    settings.process_noise_velocity = 3200.0;
    settings.measurement_noise = 18.0;
    settings.velocity_damping = 0.04;
    settings.acquisitionFrames = 3;
    settings.runtimeLatencySweepEnabled = true;
    return settings;
}

EstimatorMetrics simulateConstantVelocity(const aim::AimKalmanSettings& settings)
{
    constexpr double dtSec = 1.0 / 120.0;
    constexpr double lookaheadSec = 0.016;
    constexpr double velocityPxPerSec = 650.0;
    constexpr int frames = 120;

    aim::AimKalman2D filter;
    filter.setSettings(settings);

    double sumSquaredPredictionError = 0.0;
    double sumAbsPredictionError = 0.0;
    int samples = 0;

    for (int i = 0; i < frames; ++i)
    {
        const double t = static_cast<double>(i) * dtSec;
        const double trueX = velocityPxPerSec * t;
        const double measurementX = trueX + deterministicNoise(i);
        const auto telemetry = filter.update(measurementX, 0.0, dtSec, lookaheadSec);

        if (i < 8)
            continue;

        const double futureX = velocityPxPerSec * (t + lookaheadSec);
        const double error = telemetry.predicted_x - futureX;
        sumSquaredPredictionError += error * error;
        sumAbsPredictionError += std::abs(error);
        ++samples;
    }

    EstimatorMetrics metrics;
    metrics.predictionRmsPx = std::sqrt(sumSquaredPredictionError / static_cast<double>(std::max(1, samples)));
    metrics.predictionAverageAbsPx = sumAbsPredictionError / static_cast<double>(std::max(1, samples));
    return metrics;
}

EstimatorMetrics simulateReversal(const aim::AimKalmanSettings& settings)
{
    constexpr double dtSec = 1.0 / 120.0;
    constexpr double lookaheadSec = 0.016;
    constexpr double reversalTimeSec = 0.45;
    constexpr int frames = 120;

    aim::AimKalman2D filter;
    filter.setSettings(settings);

    double sumAbsPredictionError = 0.0;
    double maxAbsPredictionError = 0.0;
    int samples = 0;

    auto positionAt = [](double t)
    {
        constexpr double pivot = 0.45;
        constexpr double speed = 700.0;
        return t < pivot
            ? speed * t
            : speed * pivot - speed * (t - pivot);
    };

    for (int i = 0; i < frames; ++i)
    {
        const double t = static_cast<double>(i) * dtSec;
        const double measurementX = positionAt(t) + deterministicNoise(i);
        const auto telemetry = filter.update(measurementX, 0.0, dtSec, lookaheadSec);
        const double futureX = positionAt(t + lookaheadSec);
        const double error = std::abs(telemetry.predicted_x - futureX);
        maxAbsPredictionError = std::max(maxAbsPredictionError, error);

        if (t > reversalTimeSec && t < reversalTimeSec + 0.120)
        {
            sumAbsPredictionError += error;
            ++samples;
        }
    }

    EstimatorMetrics metrics;
    metrics.reversalAverageAbsPx = sumAbsPredictionError / static_cast<double>(std::max(1, samples));
    metrics.reversalMaxAbsPx = maxAbsPredictionError;
    return metrics;
}

void testStreamSharpnessConvergesWithoutSpeedCapDependency()
{
    const StepConvergenceMetrics legacy = simulateStreamStep(120.0, 18.0, 1800.0);
    const StepConvergenceMetrics tuned = simulateStreamStep(120.0, 56.0, 1800.0);

    REQUIRE(legacy.timeToTwoPxMs > 210.0);
    REQUIRE(tuned.timeToTwoPxMs > 0.0);
    REQUIRE(tuned.timeToTwoPxMs <= 115.0);
    REQUIRE(tuned.timeToOnePxMs <= 130.0);
    REQUIRE(tuned.timeToTwoPxMs < legacy.timeToTwoPxMs * 0.55);
    REQUIRE(tuned.pathLengthPx <= 120.05);
    REQUIRE(tuned.finalErrorPx < 0.001);
}

void testSeededLatencySweepImprovesPredictionConvergence()
{
    const EstimatorMetrics legacy = simulateConstantVelocity(legacyConvergenceSettings());
    const EstimatorMetrics tuned = simulateConstantVelocity(tunedConvergenceSettings());

    REQUIRE(legacy.predictionRmsPx > 8.0);
    REQUIRE(tuned.predictionRmsPx < 1.25);
    REQUIRE(tuned.predictionRmsPx < legacy.predictionRmsPx * 0.20);
    REQUIRE(tuned.predictionAverageAbsPx < legacy.predictionAverageAbsPx * 0.20);
}

void testTunedKalmanDoesNotWorsenReversalRecovery()
{
    const EstimatorMetrics legacy = simulateReversal(legacyConvergenceSettings());
    const EstimatorMetrics tuned = simulateReversal(tunedConvergenceSettings());

    REQUIRE(tuned.reversalAverageAbsPx <= legacy.reversalAverageAbsPx);
    REQUIRE(tuned.reversalMaxAbsPx <= legacy.reversalMaxAbsPx);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "stream sharpness converges without speed cap dependency", testStreamSharpnessConvergesWithoutSpeedCapDependency },
            { "seeded latency sweep improves prediction convergence", testSeededLatencySweepImprovesPredictionConvergence },
            { "tuned Kalman does not worsen reversal recovery", testTunedKalmanDoesNotWorsenReversalRecovery },
        },
        "targeting convergence");
}
