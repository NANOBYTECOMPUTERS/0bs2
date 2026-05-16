#ifndef PROJECT_0BS_BOX_2_H
#define PROJECT_0BS_BOX_2_H

#include "config.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif
#include "dml_detector.h"
#include "mouse.h"
#include "Arduino.h"
#include "detection_buffer.h"
#include "KmboxNetConnection.h"
#include "KmboxAConnection.h"
#include "Makcu.h"
#include "diagnostics/Logger.h"
#include <mutex>

extern Config config;
#ifdef USE_CUDA
extern TrtDetector trt_detector;
#endif
extern DirectMLDetector* dml_detector;
extern DetectionBuffer detectionBuffer;
extern MouseThread* globalMouseThread;
extern diagnostics::Logger appLogger;
extern Arduino* arduinoSerial;
extern RzctlMouse* razerControl;
extern KmboxNetConnection* kmboxNetSerial;
extern KmboxAConnection* kmboxASerial;
extern MakcuConnection* makcuSerial;
extern std::atomic<bool> input_method_changed;
extern std::atomic<bool> aiming;
extern std::atomic<bool> shooting;
extern std::atomic<bool> zooming;
extern std::mutex configMutex;

#endif // PROJECT_0BS_BOX_2_H
