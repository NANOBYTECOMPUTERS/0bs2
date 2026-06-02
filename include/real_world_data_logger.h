#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../mouse/BoxTarget.h"

namespace aim
{
class RealWorldDataLogger
{
public:
    ~RealWorldDataLogger()
    {
        flush();
    }

    void setEnabled(bool enabled, const std::string& outputDir, int historyLength)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const int nextHistoryLength = std::clamp(historyLength, 2, 64);
        const std::filesystem::path nextDir = outputDir.empty()
            ? std::filesystem::path("training/datasets/real_world")
            : std::filesystem::path(outputDir);

        if (!enabled)
        {
            closeSessionLocked();
            enabled_ = false;
            return;
        }

        if (!enabled_ || nextDir != outputDir_ || nextHistoryLength != historyLength_)
        {
            closeSessionLocked();
            outputDir_ = nextDir;
            historyLength_ = nextHistoryLength;
            sessionId_ = makeSessionId();
            outputPath_ = outputDir_ / ("real_world_" + sessionId_ + ".npz");
            startTime_ = std::chrono::steady_clock::now();
            enabled_ = true;
        }
    }

    void append(
        const LockedTargetInfo& lockInfo,
        const BoxTarget& target,
        const std::pair<double, double>& learnedAimOffset,
        const std::pair<double, double>& lastMouseDelta,
        int detectionResolution)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!enabled_ || lockInfo.trackHistory.empty())
            return;

        const double timestamp = secondsSinceStart();
        stateData_.insert(
            stateData_.end(),
            {
                static_cast<float>(timestamp),
                static_cast<float>(lockInfo.trackId),
                static_cast<float>(target.smoothX),
                static_cast<float>(target.smoothY),
                static_cast<float>(target.w),
                static_cast<float>(target.h),
                static_cast<float>(lockInfo.targetVelocityX),
                static_cast<float>(lockInfo.targetVelocityY),
                static_cast<float>(lockInfo.targetBoxScaleVelocity),
                static_cast<float>(std::clamp(target.confidence, 0.0, 1.0)),
                static_cast<float>(lastMouseDelta.first),
                static_cast<float>(lastMouseDelta.second),
                static_cast<float>(learnedAimOffset.first),
                static_cast<float>(learnedAimOffset.second),
                static_cast<float>(detectionResolution),
                lockInfo.observedThisFrame ? 1.0f : 0.0f,
            });

        appendHistory(lockInfo.trackHistory);
        ++sampleCount_;

        if (sampleCount_ > 0 && sampleCount_ % flushInterval_ == 0)
            flushLocked();
    }

    void flush()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        flushLocked();
    }

private:
    struct ZipEntry
    {
        std::string name;
        std::vector<std::uint8_t> data;
        std::uint32_t crc = 0;
        std::uint32_t offset = 0;
    };

    static std::string makeSessionId()
    {
        const auto now = std::chrono::system_clock::now();
        const std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_"
            << std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
        return oss.str();
    }

    double secondsSinceStart() const
    {
        const auto now = std::chrono::steady_clock::now();
        if (startTime_.time_since_epoch().count() == 0)
            return 0.0;
        return std::chrono::duration<double>(now - startTime_).count();
    }

    void appendHistory(const std::vector<std::array<float, 8>>& source)
    {
        const size_t desired = static_cast<size_t>(historyLength_);
        const size_t available = source.size();
        const size_t copyCount = std::min(desired, available);
        const std::array<float, 8> pad = available > 0 ? source.front() : std::array<float, 8>{};

        for (size_t i = copyCount; i < desired; ++i)
            historyData_.insert(historyData_.end(), pad.begin(), pad.end());

        const size_t start = available - copyCount;
        for (size_t i = start; i < available; ++i)
            historyData_.insert(historyData_.end(), source[i].begin(), source[i].end());
    }

    static void appendLe16(std::vector<std::uint8_t>& out, std::uint16_t v)
    {
        out.push_back(static_cast<std::uint8_t>(v & 0xFF));
        out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
    }

    static void appendLe32(std::vector<std::uint8_t>& out, std::uint32_t v)
    {
        for (int i = 0; i < 4; ++i)
            out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
    }

    static std::uint32_t crc32(const std::vector<std::uint8_t>& data)
    {
        std::uint32_t crc = 0xFFFFFFFFu;
        for (std::uint8_t byte : data)
        {
            crc ^= byte;
            for (int bit = 0; bit < 8; ++bit)
                crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
        return crc ^ 0xFFFFFFFFu;
    }

    static std::vector<std::uint8_t> makeNpyFloat32(const std::vector<float>& values, const std::vector<size_t>& shape)
    {
        std::ostringstream shapeText;
        shapeText << "(";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0)
                shapeText << ", ";
            shapeText << shape[i];
        }
        if (shape.size() == 1)
            shapeText << ",";
        shapeText << ")";

        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeText.str() + ", }";
        const size_t prefix = 10;
        const size_t padding = (16 - ((prefix + header.size() + 1) % 16)) % 16;
        header.append(padding, ' ');
        header.push_back('\n');

        std::vector<std::uint8_t> out;
        out.insert(out.end(), { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 });
        appendLe16(out, static_cast<std::uint16_t>(header.size()));
        out.insert(out.end(), header.begin(), header.end());
        const auto* raw = reinterpret_cast<const std::uint8_t*>(values.data());
        out.insert(out.end(), raw, raw + values.size() * sizeof(float));
        return out;
    }

    static void appendZipLocal(std::vector<std::uint8_t>& zip, ZipEntry& entry)
    {
        entry.offset = static_cast<std::uint32_t>(zip.size());
        entry.crc = crc32(entry.data);
        appendLe32(zip, 0x04034b50u);
        appendLe16(zip, 20);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe32(zip, entry.crc);
        appendLe32(zip, static_cast<std::uint32_t>(entry.data.size()));
        appendLe32(zip, static_cast<std::uint32_t>(entry.data.size()));
        appendLe16(zip, static_cast<std::uint16_t>(entry.name.size()));
        appendLe16(zip, 0);
        zip.insert(zip.end(), entry.name.begin(), entry.name.end());
        zip.insert(zip.end(), entry.data.begin(), entry.data.end());
    }

    static void appendZipCentral(std::vector<std::uint8_t>& zip, const ZipEntry& entry)
    {
        appendLe32(zip, 0x02014b50u);
        appendLe16(zip, 20);
        appendLe16(zip, 20);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe32(zip, entry.crc);
        appendLe32(zip, static_cast<std::uint32_t>(entry.data.size()));
        appendLe32(zip, static_cast<std::uint32_t>(entry.data.size()));
        appendLe16(zip, static_cast<std::uint16_t>(entry.name.size()));
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe32(zip, 0);
        appendLe32(zip, entry.offset);
        zip.insert(zip.end(), entry.name.begin(), entry.name.end());
    }

    static std::vector<std::uint8_t> buildZip(std::vector<ZipEntry> entries)
    {
        std::vector<std::uint8_t> zip;
        for (auto& entry : entries)
            appendZipLocal(zip, entry);

        const std::uint32_t centralOffset = static_cast<std::uint32_t>(zip.size());
        for (const auto& entry : entries)
            appendZipCentral(zip, entry);
        const std::uint32_t centralSize = static_cast<std::uint32_t>(zip.size()) - centralOffset;

        appendLe32(zip, 0x06054b50u);
        appendLe16(zip, 0);
        appendLe16(zip, 0);
        appendLe16(zip, static_cast<std::uint16_t>(entries.size()));
        appendLe16(zip, static_cast<std::uint16_t>(entries.size()));
        appendLe32(zip, centralSize);
        appendLe32(zip, centralOffset);
        appendLe16(zip, 0);
        return zip;
    }

    bool flushLocked()
    {
        if (sampleCount_ <= 0 || outputPath_.empty())
            return true;

        std::error_code ec;
        std::filesystem::create_directories(outputDir_, ec);
        if (ec)
            return false;

        std::vector<ZipEntry> entries;
        entries.push_back({
            "history.npy",
            makeNpyFloat32(historyData_, {
                static_cast<size_t>(sampleCount_),
                static_cast<size_t>(historyLength_),
                static_cast<size_t>(8),
            }),
        });
        entries.push_back({
            "state.npy",
            makeNpyFloat32(stateData_, {
                static_cast<size_t>(sampleCount_),
                static_cast<size_t>(16),
            }),
        });

        const auto zip = buildZip(std::move(entries));
        const auto tmpPath = outputPath_.string() + ".tmp";
        {
            std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
            if (!out.is_open())
                return false;
            out.write(reinterpret_cast<const char*>(zip.data()), static_cast<std::streamsize>(zip.size()));
            if (!out.good())
                return false;
        }
        std::filesystem::remove(outputPath_, ec);
        ec.clear();
        std::filesystem::rename(tmpPath, outputPath_, ec);
        if (ec)
            return false;
        return true;
    }

    void closeSessionLocked()
    {
        if (flushLocked())
        {
            historyData_.clear();
            stateData_.clear();
            sampleCount_ = 0;
            outputPath_.clear();
        }
    }

    bool enabled_ = false;
    int historyLength_ = 12;
    int sampleCount_ = 0;
    const int flushInterval_ = 256;
    std::filesystem::path outputDir_ = "training/datasets/real_world";
    std::filesystem::path outputPath_;
    std::string sessionId_;
    std::chrono::steady_clock::time_point startTime_ = std::chrono::steady_clock::now();
    std::vector<float> historyData_;
    std::vector<float> stateData_;
    std::mutex mutex_;
};
} // namespace aim
