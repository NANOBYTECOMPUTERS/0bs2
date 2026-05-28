#ifndef NEURAL_TARGETING_HEAD_H
#define NEURAL_TARGETING_HEAD_H

#include <memory>
#include <string>
#include <vector>

namespace aim::neural
{
class NeuralTargetingHead
{
public:
    struct Input
    {
        float center_x = 0.0f;
        float center_y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;
        float box_scale_vel = 0.0f;
        float confidence = 0.0f;
        float refinement_x = 0.0f;
        float refinement_y = 0.0f;

        std::vector<float> predicted_x;
        std::vector<float> predicted_y;
    };

    struct Output
    {
        float refinement_offset_x = 0.0f;
        float refinement_offset_y = 0.0f;
        float confidence = 0.0f;
        bool valid = false;
        double inference_ms = 0.0;
    };

    NeuralTargetingHead();
    ~NeuralTargetingHead();

    bool load(const std::string& onnx_path);
    Output compute(const Input& input);

    void setEnabled(bool enabled);
    void setMaxRefinementPx(float max_px);
    void setMaxIterations(int iterations);
    bool available() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<NeuralTargetingHead> createNeuralTargetingHead(const std::string& modelPath);

class NeuralTargetingWorker
{
public:
    struct Request
    {
        int track_id = -1;
        int frame_id = -1;
        std::string model_path;
        float max_refinement_px = 35.0f;
        int max_iterations = 2;
        NeuralTargetingHead::Input input;
    };

    struct Result
    {
        int track_id = -1;
        int frame_id = -1;
        NeuralTargetingHead::Output output;
        bool valid = false;
    };

    static NeuralTargetingWorker& instance();

    NeuralTargetingWorker();
    ~NeuralTargetingWorker();
    NeuralTargetingWorker(const NeuralTargetingWorker&) = delete;
    NeuralTargetingWorker& operator=(const NeuralTargetingWorker&) = delete;

    void submit(const Request& request);
    bool tryGet(int trackId, Result& result);
    void clear();
    void stop();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}

#endif // NEURAL_TARGETING_HEAD_H
