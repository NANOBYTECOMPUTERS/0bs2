#include "Logger.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace
{
std::string currentTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm localTime = {};
    localtime_s(&localTime, &nowTime);

    std::ostringstream out;
    out << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << millis.count();
    return out.str();
}
}

namespace diagnostics
{
Logger::Logger(bool enabled, std::string path)
    : file_logging_enabled_(enabled),
    log_file_path_(path.empty() ? "logs/0BS.log" : std::move(path))
{
}

void Logger::configure(bool enabled, const std::string& path)
{
    setFileLogging(enabled, path);
}

void Logger::setFileLogging(bool enabled, const std::string& path)
{
    std::lock_guard<std::mutex> lock(mutex_);
    file_logging_enabled_ = enabled;
    log_file_path_ = path.empty() ? "logs/0BS.log" : path;
}

void Logger::log(const std::string& message)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!file_logging_enabled_ || log_file_path_.empty())
        return;

    const std::filesystem::path path(log_file_path_);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    std::ofstream file(log_file_path_, std::ios::app);
    if (!file.is_open())
        return;

    file << currentTimestamp() << " " << message << "\n";
}

bool Logger::isFileLoggingEnabled() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return file_logging_enabled_;
}

std::string Logger::logFilePath() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return log_file_path_;
}
}
