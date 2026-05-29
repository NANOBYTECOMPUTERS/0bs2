#ifndef TEMPORAL_PREDICTOR_H
#define TEMPORAL_PREDICTOR_H

#include <memory>
#include <string>
#include <vector>

namespace aim::neural
{
constexpr int TemporalPredictorFeatureCount = 8;

class TemporalPredictor
{
public:
    struct Input
    {
        // Flattened [x, y, w, h, vx, vy, box_scale_vel, confidence] per frame.
        std::vector<float> history;
        int history_length = 12;
    };

    struct Output
    {
        std::vector<float> future_x;
        std::vector<float> future_y;
        int prediction_horizon = 16;
        float confidence = 1.0f;
        bool valid = false;
        double inference_ms = 0.0;
    };

    TemporalPredictor();
    ~TemporalPredictor();

    bool loadModel(const std::string& onnx_path);
    Output predict(const Input& input);

    void setHistoryLength(int len);
    void setPredictionHorizon(int horizon);
    bool available() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<TemporalPredictor> createTemporalPredictor(const std::string& modelPath);

class TemporalPredictionWorker
{
public:
    struct Request
    {
        int track_id = -1;
        int frame_id = -1;
        std::string model_path;
        int history_length = 12;
        int prediction_horizon = 16;
        TemporalPredictor::Input input;
    };

    struct Result
    {
        int track_id = -1;
        int frame_id = -1;
        TemporalPredictor::Output output;
        bool valid = false;
    };

    static TemporalPredictionWorker& instance();

    TemporalPredictionWorker();
    ~TemporalPredictionWorker();
    TemporalPredictionWorker(const TemporalPredictionWorker&) = delete;
    TemporalPredictionWorker& operator=(const TemporalPredictionWorker&) = delete;

    void submit(const Request& request);
    bool tryGet(int trackId, Result& result);
    void clear();
    void stop();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}

#endif // TEMPORAL_PREDICTOR_H
