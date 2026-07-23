#include "convergence_governor.h"
#include "test_harness.h"

#include <cmath>
#include <vector>

namespace
{
aim::ConvergenceGovernorSettings enabledSettings()
{
    aim::ConvergenceGovernorSettings settings;
    settings.enabled = true;
    settings.strength = 0.80;
    settings.minGain = 0.20;
    settings.maxGain = 1.12;
    return settings;
}

aim::ConvergenceGovernorContext cleanContext(double errorX)
{
    aim::ConvergenceGovernorContext context;
    context.trackId = 42;
    context.observedThisFrame = true;
    context.missedFrames = 0;
    context.activeTrackCount = 1;
    context.kalmanEnabled = true;
    context.kalmanMeasurementNoise = 18.0;
    context.kalmanVelocityDamping = 0.04;
    context.dtSec = 0.001;
    context.stateAgeMs = 2.0;
    context.distancePx = std::abs(errorX);
    context.errorX = errorX;
    context.errorY = 0.0;
    context.targetVelocityX = 120.0;
    context.targetVelocityY = 0.0;
    context.confidence = 0.92;
    return context;
}

void testDisabledGovernorIsIdentity()
{
    aim::ConvergenceGovernor governor;
    aim::ConvergenceGovernorSettings settings = enabledSettings();
    settings.enabled = false;

    const aim::ConvergenceGovernorOutput output =
        governor.evaluate(settings, cleanContext(24.0));

    REQUIRE(!output.enabled);
    REQUIRE_NEAR(output.gainScale, 1.0, 1e-9);
    REQUIRE_NEAR(output.maxStepScale, 1.0, 1e-9);
    REQUIRE_NEAR(output.brake, 0.0, 1e-9);
}

void testOvershootBrakesConvergenceGain()
{
    aim::ConvergenceGovernor governor;
    const auto settings = enabledSettings();

    const auto first = governor.evaluate(settings, cleanContext(24.0));
    governor.observeMovement(30.0, 0.0);

    const auto overshot = governor.evaluate(settings, cleanContext(-5.0));

    REQUIRE(first.enabled);
    REQUIRE(overshot.enabled);
    REQUIRE(overshot.overshootRisk > first.overshootRisk + 0.35);
    REQUIRE(overshot.brake > first.brake + 0.25);
    REQUIRE(overshot.gainScale < first.gainScale);
    REQUIRE(overshot.maxStepScale < first.maxStepScale);
}

void testOcclusionAndMultiTargetRiskDampenOutput()
{
    aim::ConvergenceGovernor governor;
    const auto settings = enabledSettings();

    const auto clean = governor.evaluate(settings, cleanContext(18.0));
    governor.reset();

    auto ambiguous = cleanContext(18.0);
    ambiguous.observedThisFrame = false;
    ambiguous.missedFrames = 2;
    ambiguous.activeTrackCount = 4;
    ambiguous.confidence = 0.48;
    ambiguous.stateAgeMs = 68.0;
    const auto risky = governor.evaluate(settings, ambiguous);

    REQUIRE(risky.lockRisk > clean.lockRisk + 0.20);
    REQUIRE(risky.gainScale < clean.gainScale);
    REQUIRE(risky.maxStepScale < clean.maxStepScale);
}

void testGovernorOutputStaysPositiveAndClamped()
{
    aim::ConvergenceGovernor governor;
    aim::ConvergenceGovernorSettings settings = enabledSettings();
    settings.strength = 1.0;
    settings.minGain = 0.35;
    settings.maxGain = 1.05;

    auto context = cleanContext(2.0);
    context.observedThisFrame = false;
    context.missedFrames = 4;
    context.activeTrackCount = 8;
    context.confidence = 0.05;
    context.kalmanMeasurementNoise = 5000.0;
    context.kalmanVelocityDamping = 3.0;
    context.targetVelocityX = 2400.0;

    governor.evaluate(settings, cleanContext(2.0));
    governor.observeMovement(9.0, 0.0);
    context.errorX = -2.0;
    const auto output = governor.evaluate(settings, context);

    REQUIRE(output.gainScale >= settings.minGain);
    REQUIRE(output.gainScale <= settings.maxGain);
    REQUIRE(output.maxStepScale >= settings.minGain);
    REQUIRE(output.maxStepScale <= settings.maxGain);
    REQUIRE(output.gainScale > 0.0);
    REQUIRE(output.maxStepScale > 0.0);
}
}

int main()
{
    return obs2test::runTests(
        {
            { "disabled governor is identity", testDisabledGovernorIsIdentity },
            { "overshoot brakes convergence gain", testOvershootBrakesConvergenceGain },
            { "occlusion and multi-target risk dampen output", testOcclusionAndMultiTargetRiskDampenOutput },
            { "governor output stays positive and clamped", testGovernorOutputStaysPositiveAndClamped },
        },
        "convergence governor");
}
