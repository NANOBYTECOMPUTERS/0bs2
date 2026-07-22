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
        "${CMAKE_CURRENT_LIST_DIR}/../modules/tensorrt"
        "${CMAKE_CURRENT_LIST_DIR}/../modules/TensorRT-11.1.0.106")
    file(GLOB versioned_tensorrt_dirs
        LIST_DIRECTORIES true
        "${CMAKE_CURRENT_LIST_DIR}/../modules/TensorRT-*")
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
