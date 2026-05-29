#ifndef NEURAL_TRACKER_H
#define NEURAL_TRACKER_H

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace aim::neural
{
constexpr std::size_t NeuralTrackerFeatureCount = 16;

struct NeuralTrackerFeatures
{
    float distanceNorm = 1.0f;
    float iou = 0.0f;
    float sizeLogRatio = 0.0f;
    float detectionConfidence = 1.0f;
    float trackConfidence = 1.0f;
    float headingAlignment = 0.0f;
    float trackMissedNorm = 0.0f;
    float trackHitsNorm = 0.0f;
    float isLocked = 0.0f;
    float classCompatible = 1.0f;
    float dt = 0.0f;
    float speedNorm = 0.0f;
    float targetSizeNorm = 0.0f;
    float pivotOffsetXNorm = 0.0f;
    float pivotOffsetYNorm = 0.0f;
    float relaxedGate = 0.0f;
};

struct NeuralTrackerResult
{
    bool valid = false;
    float neuralScore = 0.5f;
};

class INeuralTracker
{
public:
    virtual ~INeuralTracker() = default;
    virtual bool available() const = 0;
    virtual NeuralTrackerResult score(const NeuralTrackerFeatures& features) = 0;
};

std::array<float, NeuralTrackerFeatureCount> neuralTrackerFeatureArray(const NeuralTrackerFeatures& features);
std::shared_ptr<INeuralTracker> createNeuralTracker(const std::string& modelPath, const std::string& runtime);
std::shared_ptr<INeuralTracker> createOnnxNeuralTracker(const std::string& modelPath);

void logNeuralTrackerAssociation(
    const std::string& logPath,
    const NeuralTrackerFeatures& features,
    float neuralScore,
    float classicalScore,
    float finalScore,
    bool accepted,
    bool chosen);
}

#endif // NEURAL_TRACKER_H
