#pragma once

#include <NvInfer.h>
#include <NvInferVersion.h>

#include <cstdint>

namespace trt_compat
{
    inline nvinfer1::NetworkDefinitionCreationFlags explicitBatchNetworkFlags()
    {
#if NV_TENSORRT_MAJOR < 11
        return 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
        return 0U;
#endif
    }

    inline void enableFp16IfAvailable(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
    {
        if (!config)
        {
            return;
        }

#if NV_TENSORRT_MAJOR < 11
        if (!builder || builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
#else
        (void)builder;
#endif
    }
}
