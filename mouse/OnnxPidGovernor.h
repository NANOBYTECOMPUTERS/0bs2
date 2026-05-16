#ifndef ONNX_PID_GOVERNOR_H
#define ONNX_PID_GOVERNOR_H

#include "PidGovernor.h"

#include <memory>
#include <string>

namespace aim
{
std::shared_ptr<IPidGovernor> createOnnxPidGovernor(const std::string& modelPath);
}

#endif
