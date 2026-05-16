#ifndef RUNTIME_SUPERVISOR_H
#define RUNTIME_SUPERVISOR_H

#include <exception>
#include <functional>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

class RuntimeSupervisor
{
public:
    using CrashHandler = std::function<void(const char*, const std::exception*)>;

    explicit RuntimeSupervisor(CrashHandler crashHandler)
        : crashHandler_(std::move(crashHandler))
    {
    }

    RuntimeSupervisor(const RuntimeSupervisor&) = delete;
    RuntimeSupervisor& operator=(const RuntimeSupervisor&) = delete;

    ~RuntimeSupervisor()
    {
        joinAll();
    }

    template <typename Func>
    void start(const char* name, Func func)
    {
        threads_.emplace_back([this, name, func]() mutable {
            try
            {
                func();
            }
            catch (const std::exception& e)
            {
                handleCrash(name, &e);
            }
            catch (...)
            {
                handleCrash(name, nullptr);
            }
        });
    }

    void joinAll();

private:
    void handleCrash(const char* name, const std::exception* ex);

    CrashHandler crashHandler_;
    std::vector<std::thread> threads_;
};

#endif // RUNTIME_SUPERVISOR_H
