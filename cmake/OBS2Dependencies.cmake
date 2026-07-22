include_guard(GLOBAL)

function(obs2_first_existing_directory out_var)
    foreach(candidate IN LISTS ARGN)
        if(candidate AND IS_DIRECTORY "${candidate}")
            get_filename_component(resolved "${candidate}" ABSOLUTE)
            set("${out_var}" "${resolved}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set("${out_var}" "" PARENT_SCOPE)
endfunction()

function(obs2_configure_cuda_dependencies)
    find_package(CUDAToolkit REQUIRED)

    set(tensorrt_candidates
        "${OBS2_TENSORRT_ROOT}"
        "$ENV{TensorRTDir}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/tensorrt"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/TensorRT-11.1.0.106")
    file(GLOB versioned_tensorrt_dirs
        LIST_DIRECTORIES true
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/TensorRT-*")
    list(APPEND tensorrt_candidates ${versioned_tensorrt_dirs})

    obs2_first_existing_directory(OBS2_RESOLVED_TENSORRT_ROOT ${tensorrt_candidates})
    if(NOT OBS2_RESOLVED_TENSORRT_ROOT)
        message(FATAL_ERROR "TensorRT was not found. Set OBS2_TENSORRT_ROOT or TensorRTDir.")
    endif()

    find_path(OBS2_TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATHS "${OBS2_RESOLVED_TENSORRT_ROOT}/include"
        NO_DEFAULT_PATH)
    find_library(OBS2_TENSORRT_NVINFER_LIBRARY
        NAMES nvinfer_11 nvinfer
        PATHS
            "${OBS2_RESOLVED_TENSORRT_ROOT}/lib"
            "${OBS2_RESOLVED_TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH)
    find_library(OBS2_TENSORRT_ONNX_PARSER_LIBRARY
        NAMES nvonnxparser_11 nvonnxparser
        PATHS
            "${OBS2_RESOLVED_TENSORRT_ROOT}/lib"
            "${OBS2_RESOLVED_TENSORRT_ROOT}/lib/x64"
        NO_DEFAULT_PATH)

    if(NOT OBS2_TENSORRT_INCLUDE_DIR OR NOT OBS2_TENSORRT_NVINFER_LIBRARY OR NOT OBS2_TENSORRT_ONNX_PARSER_LIBRARY)
        message(FATAL_ERROR "TensorRT headers/libraries were incomplete under ${OBS2_RESOLVED_TENSORRT_ROOT}.")
    endif()

    add_library(OBS2::TensorRT INTERFACE IMPORTED)
    target_include_directories(OBS2::TensorRT INTERFACE "${OBS2_TENSORRT_INCLUDE_DIR}")
    target_link_libraries(OBS2::TensorRT
        INTERFACE
            "${OBS2_TENSORRT_NVINFER_LIBRARY}"
            "${OBS2_TENSORRT_ONNX_PARSER_LIBRARY}")

    message(STATUS "TensorRT: ${OBS2_RESOLVED_TENSORRT_ROOT}")
endfunction()

function(obs2_configure_directml_dependencies)
    message(STATUS "DirectML app target migration is staged for a later pass; current builds still use packages.config/MSBuild.")
endfunction()

function(obs2_configure_opencv_dependency)
    set(opencv_candidates
        "${OBS2_OPENCV_ROOT}"
        "$ENV{OpenCV_ROOT}"
        "$ENV{OpenCVDir}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/opencv-5.0.0/build/cuda/install")
    file(GLOB versioned_opencv_dirs
        LIST_DIRECTORIES true
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/opencv-*/build/cuda/install")
    list(APPEND opencv_candidates ${versioned_opencv_dirs})

    foreach(candidate IN LISTS opencv_candidates)
        if(NOT candidate)
            continue()
        endif()

        foreach(config_dir
            "${candidate}"
            "${candidate}/x64/vc18/lib"
            "${candidate}/x64/vc17/lib"
            "${candidate}/lib"
            "${candidate}/build")
            if(EXISTS "${config_dir}/OpenCVConfig.cmake")
                set(OpenCV_DIR "${config_dir}" CACHE PATH "OpenCV package directory" FORCE)
                break()
            endif()
        endforeach()

        if(OpenCV_DIR)
            break()
        endif()
    endforeach()

    find_package(OpenCV REQUIRED COMPONENTS core)

    if(NOT TARGET OBS2::OpenCV)
        add_library(OBS2::OpenCV INTERFACE IMPORTED)
        target_include_directories(OBS2::OpenCV INTERFACE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(OBS2::OpenCV INTERFACE ${OpenCV_LIBS})
    endif()

    set(opencv_bin_candidates
        "${OpenCV_DIR}/../../bin"
        "${OpenCV_DIR}/../../../bin"
        "${OpenCV_DIR}/../bin")
    foreach(candidate IN LISTS opencv_candidates)
        if(candidate)
            list(APPEND opencv_bin_candidates
                "${candidate}/x64/vc18/bin"
                "${candidate}/x64/vc17/bin"
                "${candidate}/bin")
        endif()
    endforeach()

    obs2_first_existing_directory(OBS2_RESOLVED_OPENCV_BIN_DIR ${opencv_bin_candidates})
    set(OBS2_OPENCV_BIN_DIR "${OBS2_RESOLVED_OPENCV_BIN_DIR}" PARENT_SCOPE)

    set(cuda_runtime_bin_candidates
        "$ENV{CUDA_PATH}/bin/x64"
        "$ENV{CUDA_PATH}/bin"
        "${CUDAToolkit_BIN_DIR}/x64"
        "${CUDAToolkit_BIN_DIR}")
    file(GLOB cuda_versioned_bin_dirs
        LIST_DIRECTORIES true
        "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/x64"
        "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA/v*/bin")
    list(SORT cuda_versioned_bin_dirs COMPARE NATURAL ORDER DESCENDING)
    list(APPEND cuda_runtime_bin_candidates ${cuda_versioned_bin_dirs})
    obs2_first_existing_directory(OBS2_RESOLVED_CUDA_RUNTIME_BIN_DIR ${cuda_runtime_bin_candidates})
    set(OBS2_CUDA_RUNTIME_BIN_DIR "${OBS2_RESOLVED_CUDA_RUNTIME_BIN_DIR}" PARENT_SCOPE)

    set(cudnn_runtime_bin_candidates
        "$ENV{CUDNN_ROOT}/bin/x64"
        "$ENV{CUDNN_ROOT}/bin"
        "$ENV{CUDNN_PATH}/bin/x64"
        "$ENV{CUDNN_PATH}/bin")
    file(GLOB cudnn_versioned_bin_dirs
        LIST_DIRECTORIES true
        "$ENV{ProgramFiles}/NVIDIA/CUDNN/v*/bin/*/x64"
        "$ENV{ProgramFiles}/NVIDIA/CUDNN/v*/bin/*")
    list(SORT cudnn_versioned_bin_dirs COMPARE NATURAL ORDER DESCENDING)
    list(APPEND cudnn_runtime_bin_candidates
        ${cudnn_versioned_bin_dirs}
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../modules/libtorch/lib")
    obs2_first_existing_directory(OBS2_RESOLVED_CUDNN_RUNTIME_BIN_DIR ${cudnn_runtime_bin_candidates})
    set(OBS2_CUDNN_RUNTIME_BIN_DIR "${OBS2_RESOLVED_CUDNN_RUNTIME_BIN_DIR}" PARENT_SCOPE)

    message(STATUS "OpenCV: ${OpenCV_DIR}")
    if(OBS2_RESOLVED_OPENCV_BIN_DIR)
        message(STATUS "OpenCV runtime bin: ${OBS2_RESOLVED_OPENCV_BIN_DIR}")
    endif()
    if(OBS2_RESOLVED_CUDA_RUNTIME_BIN_DIR)
        message(STATUS "CUDA runtime bin: ${OBS2_RESOLVED_CUDA_RUNTIME_BIN_DIR}")
    endif()
    if(OBS2_RESOLVED_CUDNN_RUNTIME_BIN_DIR)
        message(STATUS "cuDNN runtime bin: ${OBS2_RESOLVED_CUDNN_RUNTIME_BIN_DIR}")
    endif()
endfunction()

function(obs2_copy_opencv_runtime target_name)
    if(NOT WIN32 OR NOT OBS2_OPENCV_BIN_DIR)
        return()
    endif()

    file(GLOB opencv_runtime_dlls
        "${OBS2_OPENCV_BIN_DIR}/opencv_world*.dll"
        "${OBS2_OPENCV_BIN_DIR}/opencv_core*.dll")

    foreach(dll IN LISTS opencv_runtime_dlls)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${dll}"
                "$<TARGET_FILE_DIR:${target_name}>"
            VERBATIM)
    endforeach()
endfunction()

function(obs2_apply_opencv_test_environment test_name)
    foreach(runtime_dir
        "${OBS2_OPENCV_BIN_DIR}"
        "${OBS2_CUDA_RUNTIME_BIN_DIR}"
        "${OBS2_CUDNN_RUNTIME_BIN_DIR}")
        if(runtime_dir)
            set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:${runtime_dir}")
        endif()
    endforeach()
endfunction()
