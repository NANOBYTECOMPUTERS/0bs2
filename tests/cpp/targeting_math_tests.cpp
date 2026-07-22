#include "aim_imm.h"
#include "aim_kalman.h"
#include "ego_motion_compensator.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
class TestFailure : public std::exception
{
public:
    explicit TestFailure(std::string message)
        : message_(std::move(message))
    {
    }

    const char* what() const noexcept override
    {
        return message_.c_str();
    }

private:
    std::string message_;
};

void require(bool condition, const char* expression, const char* file, int line)
{
    if (condition)
        return;

    std::ostringstream oss;
    oss << file << ":" << line << ": requirement failed: " << expression;
    throw TestFailure(oss.str());
}

void requireNear(double actual, double expected, double tolerance, const char* expression, const char* file, int line)
{
    if (std::isfinite(actual) && std::fabs(actual - expected) <= tolerance)
        return;

    std::ostringstream oss;
    oss << file << ":" << line << ": " << expression << " expected " << expected
        << " +/- " << tolerance << ", got " << actual;
    throw TestFailure(oss.str());
}

#define REQUIRE(expr) require((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_NEAR(actual, expected, tolerance) requireNear((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

void testKalmanSettingsClamp()
{
    aim::AimKalman2D filter;
    aim::AimKalmanSettings settings;
    settings.process_noise_position = -10.0;
    settings.process_noise_velocity = 1000000.0;
    settings.measurement_noise = 0.0;
    settings.velocity_damping = -1.0;
    settings.max_velocity = 2.0;
    settings.warmup_frames = 99;
    settings.acquisitionFrames = 99;

    filter.setSettings(settings);
    const auto& clamped = filter.settings();
    REQUIRE_NEAR(clamped.process_noise_position, 1e-4, 1e-9);
    REQUIRE_NEAR(clamped.process_noise_velocity, 50000.0, 1e-9);
    REQUIRE_NEAR(clamped.measurement_noise, 1e-4, 1e-9);
    REQUIRE_NEAR(clamped.velocity_damping, 0.0, 1e-9);
    REQUIRE_NEAR(clamped.max_velocity, 100.0, 1e-9);
    REQUIRE(clamped.warmup_frames == 20);
    REQUIRE(clamped.acquisitionFrames == 5);
}

void testKalmanDisabledTracksMeasurementsAndPredicts()
{
    aim::AimKalman2D filter;
    aim::AimKalmanSettings settings;
    settings.enabled = false;
    settings.warmup_frames = 0;
    settings.velocity_damping = 0.0;
    settings.max_velocity = 10000.0;
    filter.setSettings(settings);
    filter.reset();

    filter.update(10.0, 20.0, 0.016, 0.0);
    const auto telemetry = filter.update(14.0, 28.0, 0.020, 0.100);

    REQUIRE(telemetry.initialized);
    REQUIRE(!telemetry.enabled);
    REQUIRE_NEAR(telemetry.estimate_x, 14.0, 1e-9);
    REQUIRE_NEAR(telemetry.estimate_y, 28.0, 1e-9);
    REQUIRE_NEAR(telemetry.velocity_x, 200.0, 1e-6);
    REQUIRE_NEAR(telemetry.velocity_y, 400.0, 1e-6);
    REQUIRE(telemetry.predicted_x > telemetry.estimate_x);
    REQUIRE(telemetry.predicted_y > telemetry.estimate_y);
}

void testKalmanRuntimeSweepPredictionRamp()
{
    aim::AimKalman2D filter;
    aim::AimKalmanSettings settings;
    settings.runtimeLatencySweepEnabled = true;
    settings.velocitySeedEnabled = true;
    settings.acquisitionFrames = 4;
    settings.warmup_frames = 0;
    settings.velocity_damping = 0.0;
    settings.max_velocity = 10000.0;
    filter.setSettings(settings);
    filter.reset();

    const auto first = filter.update(0.0, 0.0, 0.016, 0.050);
    const auto second = filter.update(10.0, 0.0, 0.016, 0.050);
    const auto third = filter.update(20.0, 0.0, 0.016, 0.050);

    REQUIRE_NEAR(first.prediction_weight, 0.0, 1e-9);
    REQUIRE_NEAR(second.prediction_weight, 0.25, 1e-9);
    REQUIRE_NEAR(third.prediction_weight, 0.50, 1e-9);
    REQUIRE(second.predicted_x >= second.estimate_x);
    REQUIRE(third.predicted_x >= third.estimate_x);
}

void testImmProducesFiniteProbabilitiesAndPrediction()
{
    aim::AimIMM2D filter;
    aim::AimKalmanSettings settings;
    settings.max_velocity = 20000.0;
    settings.velocity_damping = 0.02;
    filter.setSettings(settings);
    filter.reset();

    for (int i = 0; i < 12; ++i)
    {
        const double x = static_cast<double>(i) * 7.5;
        const double y = static_cast<double>(i) * 2.0;
        const auto telemetry = filter.update(x, y, 0.016, 0.030);
        REQUIRE(std::isfinite(telemetry.estimate_x));
        REQUIRE(std::isfinite(telemetry.estimate_y));
        REQUIRE(std::isfinite(telemetry.velocity_x));
        REQUIRE(std::isfinite(telemetry.velocity_y));
    }

    const auto probabilities = filter.modeProbabilities();
    REQUIRE(probabilities.constant_velocity >= 0.0);
    REQUIRE(probabilities.constant_acceleration >= 0.0);
    REQUIRE_NEAR(probabilities.constant_velocity + probabilities.constant_acceleration, 1.0, 1e-9);

    const auto position = filter.position();
    const auto predicted = filter.predict(0.050);
    REQUIRE(predicted.first >= position.first);
    REQUIRE(predicted.second >= position.second);
}

void testEgoMotionConsumesTimeWindowAndClamps()
{
    aim::EgoMotionCompensator compensator;
    aim::EgoMotionSettings settings;
    settings.enabled = true;
    settings.strength = 0.5;
    settings.maxShiftPx = 6.0;
    settings.maxAgeMs = 120;
    compensator.setSettings(settings);

    const auto base = aim::EgoMotionCompensator::Clock::now();
    compensator.recordDelta(4.0, 0.0, base + std::chrono::milliseconds(10));
    compensator.recordDelta(20.0, 0.0, base + std::chrono::milliseconds(30));

    const auto first = compensator.consume(
        base,
        base + std::chrono::milliseconds(20),
        base + std::chrono::milliseconds(20));
    REQUIRE(first.valid);
    REQUIRE_NEAR(first.dx, 2.0, 1e-9);
    REQUIRE_NEAR(first.dy, 0.0, 1e-9);

    const auto second = compensator.consume(
        base + std::chrono::milliseconds(20),
        base + std::chrono::milliseconds(40),
        base + std::chrono::milliseconds(40));
    REQUIRE(second.valid);
    REQUIRE_NEAR(std::hypot(second.dx, second.dy), 6.0, 1e-9);
}

void testEgoMotionDisabledClearsSamples()
{
    aim::EgoMotionCompensator compensator;
    aim::EgoMotionSettings settings;
    settings.enabled = true;
    compensator.setSettings(settings);

    const auto base = aim::EgoMotionCompensator::Clock::now();
    compensator.recordDelta(5.0, 0.0, base + std::chrono::milliseconds(1));

    settings.enabled = false;
    compensator.setSettings(settings);
    const auto shift = compensator.consume(
        base,
        base + std::chrono::milliseconds(10),
        base + std::chrono::milliseconds(10));
    REQUIRE(!shift.valid);
    REQUIRE_NEAR(shift.dx, 0.0, 1e-9);
    REQUIRE_NEAR(shift.dy, 0.0, 1e-9);
}

struct NamedTest
{
    const char* name;
    void (*run)();
};
}

int main()
{
    const std::vector<NamedTest> tests = {
        { "Kalman settings clamp", testKalmanSettingsClamp },
        { "Kalman disabled tracks measurements and predicts", testKalmanDisabledTracksMeasurementsAndPredicts },
        { "Kalman runtime sweep prediction ramp", testKalmanRuntimeSweepPredictionRamp },
        { "IMM finite probabilities and prediction", testImmProducesFiniteProbabilitiesAndPrediction },
        { "Ego motion consumes time window and clamps", testEgoMotionConsumesTimeWindowAndClamps },
        { "Ego motion disabled clears samples", testEgoMotionDisabledClearsSamples },
    };

    int failures = 0;
    for (const auto& test : tests)
    {
        try
        {
            test.run();
            std::cout << "[PASS] " << test.name << "\n";
        }
        catch (const std::exception& e)
        {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " native targeting test(s) failed.\n";
        return 1;
    }

    std::cout << tests.size() << " native targeting tests passed.\n";
    return 0;
}
