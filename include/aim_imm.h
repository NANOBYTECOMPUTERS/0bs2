#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "aim_kalman.h"

namespace aim
{
class AimIMM2D
{
public:
    struct ModeProbability
    {
        double constant_velocity = 0.5;
        double constant_acceleration = 0.5;
    };

    AimIMM2D() = default;

    void setSettings(const AimKalmanSettings& settings)
    {
        settings_ = clampSettings(settings);
    }

    const AimKalmanSettings& settings() const
    {
        return settings_;
    }

    void reset()
    {
        modes_[0] = ModeState(false);
        modes_[1] = ModeState(true);
        combinedX_ = AxisState();
        combinedY_ = AxisState();
        modeProbability_ = { 0.72, 0.28 };
        initialized_ = false;
        hasLastMeasurement_ = false;
        lastMeasurementX_ = 0.0;
        lastMeasurementY_ = 0.0;
    }

    bool initialized() const
    {
        return initialized_;
    }

    std::pair<double, double> position() const
    {
        return { combinedX_.position, combinedY_.position };
    }

    std::pair<double, double> velocity() const
    {
        return { combinedX_.velocity, combinedY_.velocity };
    }

    ModeProbability modeProbabilities() const
    {
        return {
            modeProbability_[0],
            modeProbability_[1],
        };
    }

    void applyPositionOffset(double dx, double dy)
    {
        if (!initialized_ || (!std::isfinite(dx) || !std::isfinite(dy)))
            return;

        for (auto& mode : modes_)
        {
            mode.x.position += dx;
            mode.y.position += dy;
        }
        combinedX_.position += dx;
        combinedY_.position += dy;
        if (hasLastMeasurement_)
        {
            lastMeasurementX_ += dx;
            lastMeasurementY_ += dy;
        }
    }

    std::pair<double, double> predict(double lookaheadSec) const
    {
        if (!initialized_)
            return {};

        const double lookahead = std::clamp(lookaheadSec, 0.0, 1.5);
        if (lookahead <= 0.0)
            return position();

        double x = 0.0;
        double y = 0.0;
        for (size_t i = 0; i < modes_.size(); ++i)
        {
            x += modeProbability_[i] * predictAxisAhead(modes_[i].x, modes_[i].constant_acceleration, lookahead);
            y += modeProbability_[i] * predictAxisAhead(modes_[i].y, modes_[i].constant_acceleration, lookahead);
        }
        return { x, y };
    }

    AimKalmanTelemetry update(double measurementX, double measurementY, double dt, double lookaheadSec)
    {
        AimKalmanTelemetry telemetry;
        telemetry.enabled = settings_.enabled;
        telemetry.measurement_x = measurementX;
        telemetry.measurement_y = measurementY;

        const double clampedDt = std::clamp(dt, 1e-4, 0.25);
        const double lookahead = std::clamp(lookaheadSec, 0.0, 1.5);
        telemetry.dt = clampedDt;

        if (!initialized_)
        {
            initialize(measurementX, measurementY);
            return buildTelemetry(measurementX, measurementY, 0.0, 0.0, clampedDt, lookahead);
        }

        if (!settings_.enabled)
        {
            const double prevX = combinedX_.position;
            const double prevY = combinedY_.position;
            combinedX_.position = measurementX;
            combinedY_.position = measurementY;
            combinedX_.velocity = clampAbs((measurementX - lastMeasurementX_) / clampedDt, settings_.max_velocity);
            combinedY_.velocity = clampAbs((measurementY - lastMeasurementY_) / clampedDt, settings_.max_velocity);
            combinedX_.acceleration = 0.0;
            combinedY_.acceleration = 0.0;
            for (auto& mode : modes_)
            {
                mode.x = combinedX_;
                mode.y = combinedY_;
            }
            lastMeasurementX_ = measurementX;
            lastMeasurementY_ = measurementY;
            hasLastMeasurement_ = true;
            return buildTelemetry(measurementX, measurementY, measurementX - prevX, measurementY - prevY, clampedDt, lookahead);
        }

        mixModes();

        std::array<double, 2> logScore{};
        std::array<double, 2> innovationX{};
        std::array<double, 2> innovationY{};
        for (size_t i = 0; i < modes_.size(); ++i)
        {
            predictMode(modes_[i], clampedDt);
            innovationX[i] = updateAxis(modes_[i].x, measurementX);
            innovationY[i] = updateAxis(modes_[i].y, measurementY);
            const double likelihoodX = measurementLikelihood(innovationX[i], modes_[i].x.lastInnovationVariance);
            const double likelihoodY = measurementLikelihood(innovationY[i], modes_[i].y.lastInnovationVariance);
            const double predictedModeProb = std::max(1e-12, predictedModeProbability_[i]);
            logScore[i] = std::log(predictedModeProb) + std::log(likelihoodX) + std::log(likelihoodY);
        }

        normalizeModeProbabilities(logScore);
        combineModes();

        lastMeasurementX_ = measurementX;
        lastMeasurementY_ = measurementY;
        hasLastMeasurement_ = true;

        const double blendedInnovationX =
            modeProbability_[0] * innovationX[0] + modeProbability_[1] * innovationX[1];
        const double blendedInnovationY =
            modeProbability_[0] * innovationY[0] + modeProbability_[1] * innovationY[1];
        return buildTelemetry(measurementX, measurementY, blendedInnovationX, blendedInnovationY, clampedDt, lookahead);
    }

private:
    struct AxisState
    {
        double position = 0.0;
        double velocity = 0.0;
        double acceleration = 0.0;
        double covariance[3][3] = {
            { 150.0, 0.0, 0.0 },
            { 0.0, 700.0, 0.0 },
            { 0.0, 0.0, 700.0 },
        };
        double lastInnovationVariance = 1.0;
    };

    struct ModeState
    {
        explicit ModeState(bool accelerationMode = false)
            : constant_acceleration(accelerationMode)
        {
        }

        AxisState x;
        AxisState y;
        bool constant_acceleration = false;
    };

    static AimKalmanSettings clampSettings(const AimKalmanSettings& in)
    {
        AimKalmanSettings out = in;
        out.process_noise_position = std::clamp(out.process_noise_position, 1e-4, 5000.0);
        out.process_noise_velocity = std::clamp(out.process_noise_velocity, 1e-4, 50000.0);
        out.measurement_noise = std::clamp(out.measurement_noise, 1e-4, 5000.0);
        out.velocity_damping = std::clamp(out.velocity_damping, 0.0, 3.0);
        out.max_velocity = std::clamp(out.max_velocity, 100.0, 60000.0);
        return out;
    }

    static double clampAbs(double value, double maxAbs)
    {
        return std::clamp(value, -maxAbs, maxAbs);
    }

    static double measurementLikelihood(double innovation, double variance)
    {
        const double s = std::max(1e-6, variance);
        constexpr double twoPi = 6.28318530717958647692;
        const double density = std::exp(-0.5 * innovation * innovation / s) / std::sqrt(twoPi * s);
        return std::clamp(density, 1e-12, 1e12);
    }

    static double predictAxisAhead(const AxisState& axis, bool constantAcceleration, double lookahead)
    {
        if (lookahead <= 0.0)
            return axis.position;
        if (constantAcceleration)
            return axis.position + axis.velocity * lookahead + 0.5 * axis.acceleration * lookahead * lookahead;
        return axis.position + axis.velocity * lookahead;
    }

    void initialize(double measurementX, double measurementY)
    {
        modeProbability_ = { 0.72, 0.28 };
        predictedModeProbability_ = modeProbability_;
        modes_[0] = ModeState(false);
        modes_[1] = ModeState(true);

        for (auto& mode : modes_)
        {
            initializeAxis(mode.x, measurementX);
            initializeAxis(mode.y, measurementY);
        }

        combineModes();
        initialized_ = true;
        hasLastMeasurement_ = true;
        lastMeasurementX_ = measurementX;
        lastMeasurementY_ = measurementY;
    }

    void initializeAxis(AxisState& axis, double measurement)
    {
        axis.position = measurement;
        axis.velocity = 0.0;
        axis.acceleration = 0.0;
        axis.covariance[0][0] = settings_.measurement_noise;
        axis.covariance[0][1] = axis.covariance[0][2] = 0.0;
        axis.covariance[1][0] = 0.0;
        axis.covariance[1][1] = settings_.process_noise_velocity;
        axis.covariance[1][2] = 0.0;
        axis.covariance[2][0] = axis.covariance[2][1] = 0.0;
        axis.covariance[2][2] = std::max(50.0, settings_.process_noise_velocity * 1.35);
        axis.lastInnovationVariance = settings_.measurement_noise;
    }

    void mixModes()
    {
        constexpr double transition_probability = 0.035;
        const double stay = 1.0 - transition_probability;
        const double transition[2][2] = {
            { stay, transition_probability },
            { transition_probability, stay },
        };

        for (size_t j = 0; j < modes_.size(); ++j)
        {
            predictedModeProbability_[j] =
                modeProbability_[0] * transition[0][j] +
                modeProbability_[1] * transition[1][j];
        }

        const auto previousModes = modes_;
        for (size_t j = 0; j < modes_.size(); ++j)
        {
            double weights[2]{ 0.5, 0.5 };
            if (predictedModeProbability_[j] > 1e-12)
            {
                weights[0] = modeProbability_[0] * transition[0][j] / predictedModeProbability_[j];
                weights[1] = modeProbability_[1] * transition[1][j] / predictedModeProbability_[j];
            }
            mixAxis(previousModes[0].x, previousModes[1].x, weights, modes_[j].x);
            mixAxis(previousModes[0].y, previousModes[1].y, weights, modes_[j].y);
        }
    }

    static void mixAxis(const AxisState& a, const AxisState& b, const double weights[2], AxisState& out)
    {
        const AxisState* states[2] = { &a, &b };
        const double mean[3] = {
            weights[0] * a.position + weights[1] * b.position,
            weights[0] * a.velocity + weights[1] * b.velocity,
            weights[0] * a.acceleration + weights[1] * b.acceleration,
        };

        out.position = mean[0];
        out.velocity = mean[1];
        out.acceleration = mean[2];

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                out.covariance[r][c] = 0.0;
                for (int i = 0; i < 2; ++i)
                {
                    const double vector[3] = {
                        states[i]->position - mean[0],
                        states[i]->velocity - mean[1],
                        states[i]->acceleration - mean[2],
                    };
                    out.covariance[r][c] += weights[i] * (states[i]->covariance[r][c] + vector[r] * vector[c]);
                }
            }
        }
        out.lastInnovationVariance = weights[0] * a.lastInnovationVariance + weights[1] * b.lastInnovationVariance;
    }

    void predictMode(ModeState& mode, double dt) const
    {
        predictAxis(mode.x, mode.constant_acceleration, dt);
        predictAxis(mode.y, mode.constant_acceleration, dt);
    }

    void predictAxis(AxisState& axis, bool constantAcceleration, double dt) const
    {
        const double damping = std::exp(-settings_.velocity_damping * dt);
        const double f[3][3] = {
            { 1.0, dt, constantAcceleration ? 0.5 * dt * dt : 0.0 },
            { 0.0, damping, constantAcceleration ? dt : 0.0 },
            { 0.0, 0.0, constantAcceleration ? 0.92 : 0.0 },
        };

        const double state[3] = { axis.position, axis.velocity, axis.acceleration };
        axis.position = f[0][0] * state[0] + f[0][1] * state[1] + f[0][2] * state[2];
        axis.velocity = clampAbs(f[1][0] * state[0] + f[1][1] * state[1] + f[1][2] * state[2], settings_.max_velocity);
        axis.acceleration = f[2][0] * state[0] + f[2][1] * state[1] + f[2][2] * state[2];

        double fp[3][3]{};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k)
                    fp[r][c] += f[r][k] * axis.covariance[k][c];

        double fpfT[3][3]{};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k)
                    fpfT[r][c] += fp[r][k] * f[c][k];

        const double posNoise = settings_.process_noise_position * dt;
        const double velNoise = settings_.process_noise_velocity * dt;
        const double accNoise = settings_.process_noise_velocity * (constantAcceleration ? 6.0 : 0.015) * dt;
        fpfT[0][0] += posNoise;
        fpfT[1][1] += velNoise;
        fpfT[2][2] += std::max(1e-6, accNoise);

        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                axis.covariance[r][c] = fpfT[r][c];

        if (!constantAcceleration)
            axis.acceleration = 0.0;
    }

    double updateAxis(AxisState& axis, double measurement) const
    {
        const double innovation = measurement - axis.position;
        const double s = std::max(1e-9, axis.covariance[0][0] + settings_.measurement_noise);
        axis.lastInnovationVariance = s;

        const double k[3] = {
            axis.covariance[0][0] / s,
            axis.covariance[1][0] / s,
            axis.covariance[2][0] / s,
        };

        axis.position += k[0] * innovation;
        axis.velocity = clampAbs(axis.velocity + k[1] * innovation, settings_.max_velocity);
        axis.acceleration += k[2] * innovation;

        const double oldP[3][3] = {
            { axis.covariance[0][0], axis.covariance[0][1], axis.covariance[0][2] },
            { axis.covariance[1][0], axis.covariance[1][1], axis.covariance[1][2] },
            { axis.covariance[2][0], axis.covariance[2][1], axis.covariance[2][2] },
        };
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                axis.covariance[r][c] = oldP[r][c] - k[r] * oldP[0][c];
                if (r == c)
                    axis.covariance[r][c] = std::max(1e-9, axis.covariance[r][c]);
            }
        }
        return innovation;
    }

    void normalizeModeProbabilities(const std::array<double, 2>& logScore)
    {
        const double maxLog = std::max(logScore[0], logScore[1]);
        const double e0 = std::exp(logScore[0] - maxLog);
        const double e1 = std::exp(logScore[1] - maxLog);
        const double sum = std::max(1e-12, e0 + e1);
        modeProbability_[0] = std::clamp(e0 / sum, 0.02, 0.98);
        modeProbability_[1] = std::clamp(e1 / sum, 0.02, 0.98);
        const double renorm = std::max(1e-12, modeProbability_[0] + modeProbability_[1]);
        modeProbability_[0] /= renorm;
        modeProbability_[1] /= renorm;
    }

    void combineModes()
    {
        combineAxis(modes_[0].x, modes_[1].x, combinedX_);
        combineAxis(modes_[0].y, modes_[1].y, combinedY_);
    }

    void combineAxis(const AxisState& cv, const AxisState& ca, AxisState& out) const
    {
        const double w0 = modeProbability_[0];
        const double w1 = modeProbability_[1];
        out.position = w0 * cv.position + w1 * ca.position;
        out.velocity = clampAbs(w0 * cv.velocity + w1 * ca.velocity, settings_.max_velocity);
        out.acceleration = w0 * cv.acceleration + w1 * ca.acceleration;

        const AxisState* states[2] = { &cv, &ca };
        const double weights[2] = { w0, w1 };
        const double mean[3] = { out.position, out.velocity, out.acceleration };
        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                out.covariance[r][c] = 0.0;
                for (int i = 0; i < 2; ++i)
                {
                    const double vector[3] = {
                        states[i]->position - mean[0],
                        states[i]->velocity - mean[1],
                        states[i]->acceleration - mean[2],
                    };
                    out.covariance[r][c] += weights[i] * (states[i]->covariance[r][c] + vector[r] * vector[c]);
                }
            }
        }
        out.lastInnovationVariance = w0 * cv.lastInnovationVariance + w1 * ca.lastInnovationVariance;
    }

    AimKalmanTelemetry buildTelemetry(
        double measurementX,
        double measurementY,
        double innovationX,
        double innovationY,
        double dt,
        double lookaheadSec) const
    {
        AimKalmanTelemetry telemetry;
        telemetry.initialized = initialized_;
        telemetry.enabled = settings_.enabled;
        telemetry.dt = dt;
        telemetry.warmup_remaining = 0;
        telemetry.measurement_x = measurementX;
        telemetry.measurement_y = measurementY;
        telemetry.estimate_x = combinedX_.position;
        telemetry.estimate_y = combinedY_.position;
        telemetry.velocity_x = combinedX_.velocity;
        telemetry.velocity_y = combinedY_.velocity;
        telemetry.innovation_x = innovationX;
        telemetry.innovation_y = innovationY;
        telemetry.prediction_weight = initialized_ ? 1.0 : 0.0;
        const auto future = predict(lookaheadSec);
        telemetry.predicted_x = initialized_ ? future.first : combinedX_.position;
        telemetry.predicted_y = initialized_ ? future.second : combinedY_.position;
        return telemetry;
    }

private:
    AimKalmanSettings settings_{};
    std::array<ModeState, 2> modes_{ ModeState(false), ModeState(true) };
    std::array<double, 2> modeProbability_{ 0.72, 0.28 };
    std::array<double, 2> predictedModeProbability_{ 0.72, 0.28 };
    AxisState combinedX_{};
    AxisState combinedY_{};
    bool initialized_ = false;
    bool hasLastMeasurement_ = false;
    double lastMeasurementX_ = 0.0;
    double lastMeasurementY_ = 0.0;
};
} // namespace aim
