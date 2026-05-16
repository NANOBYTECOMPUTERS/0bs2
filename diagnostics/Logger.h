#ifndef DIAGNOSTICS_LOGGER_H
#define DIAGNOSTICS_LOGGER_H

#include <mutex>
#include <string>

namespace diagnostics
{
class Logger
{
public:
    Logger() = default;
    explicit Logger(bool enabled, std::string path = "logs/0BS.log");

    void configure(bool enabled, const std::string& path);
    void setFileLogging(bool enabled, const std::string& path);
    void log(const std::string& message);

    bool isFileLoggingEnabled() const;
    std::string logFilePath() const;

private:
    mutable std::mutex mutex_;
    bool file_logging_enabled_ = false;
    std::string log_file_path_ = "logs/0BS.log";
};
}

#endif // DIAGNOSTICS_LOGGER_H
