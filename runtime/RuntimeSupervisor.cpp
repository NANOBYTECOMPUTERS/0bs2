#include "RuntimeSupervisor.h"

void RuntimeSupervisor::joinAll()
{
    for (auto& thread : threads_)
    {
        if (thread.joinable())
            thread.join();
    }
}

void RuntimeSupervisor::handleCrash(const char* name, const std::exception* ex)
{
    if (crashHandler_)
    {
        crashHandler_(name, ex);
        return;
    }

    std::cerr << "[Thread] " << name << " crashed: "
              << (ex ? ex->what() : "unknown exception") << std::endl;
}
