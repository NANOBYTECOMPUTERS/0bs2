#ifndef SMART_BLENDER_H
#define SMART_BLENDER_H

namespace aim
{
struct SmartBlenderSettings
{
    bool enabled = false;
    double aggression = 0.65;
    double nearTargetDamping = 0.75;
    double deadzonePx = 0.0;
    double jerkLimitPx = 0.65;
    double confidenceFloor = 0.45;
    double speedReferencePxPerSec = 1100.0;
};

struct SmartBlendInput
{
    double desiredX = 0.0;
    double desiredY = 0.0;
    double distance = 0.0;
    double precisionRadius = 0.0;
    double targetSize = 48.0;
    double targetSpeed = 0.0;
    double confidence = 1.0;
    double dtSeconds = 1.0 / 240.0;
};

struct SmartBlendOutput
{
    double x = 0.0;
    double y = 0.0;
    double alpha = 1.0;
    double jerkLimitPx = 0.0;
    double nearAmount = 0.0;
    bool active = false;
};

class SmartBlender
{
public:
    void setSettings(const SmartBlenderSettings& nextSettings);
    void reset();
    SmartBlendOutput apply(const SmartBlendInput& input);

private:
    SmartBlenderSettings settings;
    double previousX = 0.0;
    double previousY = 0.0;
    bool hasPrevious = false;

    static double clampFinite(double value, double lo, double hi, double fallback);
    static double smoothStep(double value);
};
}

#endif
