cmake_minimum_required(VERSION 3.28)

project(OBS2
    VERSION 0.1.0
    DESCRIPTION "Parallel CMake modernization layer for 0BS"
    LANGUAGES CXX)

option(OBS2_BUILD_NATIVE_TESTS "Build native C++ tests for pure targeting logic." ON)
option(OBS2_ENABLE_CLANG_TIDY "Run clang-tidy for CMake-built C++ targets when available." OFF)
option(OBS2_USE_CUDA "Reserve CUDA dependency discovery for future migrated targets." OFF)
option(OBS2_USE_DIRECTML "Reserve DirectML dependency discovery for future migrated targets." OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(OBS2Dependencies)

if(OBS2_ENABLE_CLANG_TIDY)
    find_program(OBS2_CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${OBS2_CLANG_TIDY_EXE}")
endif()

add_library(obs2_targeting_core INTERFACE)
target_compile_features(obs2_targeting_core INTERFACE cxx_std_20)
target_include_directories(obs2_targeting_core
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    target_compile_options(obs2_targeting_core
        INTERFACE
            /permissive-
            /W4
            /Zc:__cplusplus)
else()
    target_compile_options(obs2_targeting_core
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic)
endif()

if(OBS2_USE_CUDA)
    obs2_configure_cuda_dependencies()
endif()

if(OBS2_USE_DIRECTML)
    obs2_configure_directml_dependencies()
endif()

if(OBS2_BUILD_NATIVE_TESTS)
    enable_testing()
    obs2_configure_opencv_dependency()

    add_executable(obs2_targeting_tests
        tests/cpp/targeting_math_tests.cpp)
    target_link_libraries(obs2_targeting_tests
        PRIVATE
            obs2_targeting_core)

    add_test(NAME obs2_targeting_tests COMMAND obs2_targeting_tests)

    add_executable(obs2_capture_geometry_tests
        tests/cpp/capture_geometry_tests.cpp)
    target_include_directories(obs2_capture_geometry_tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(obs2_capture_geometry_tests
        PRIVATE
            obs2_targeting_core
            OBS2::OpenCV)
    obs2_copy_opencv_runtime(obs2_capture_geometry_tests)
    add_test(NAME obs2_capture_geometry_tests COMMAND obs2_capture_geometry_tests)
    obs2_apply_opencv_test_environment(obs2_capture_geometry_tests)

    add_executable(obs2_postprocess_tests
        tests/cpp/postprocess_tests.cpp
        detector/postProcess.cpp)
    target_compile_definitions(obs2_postprocess_tests
        PRIVATE
            USE_CUDA
            YOLO_ANNOTATION_WORKER)
    target_include_directories(obs2_postprocess_tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/detector")
    target_link_libraries(obs2_postprocess_tests
        PRIVATE
            obs2_targeting_core
            OBS2::OpenCV)
    obs2_copy_opencv_runtime(obs2_postprocess_tests)
    add_test(NAME obs2_postprocess_tests COMMAND obs2_postprocess_tests)
    obs2_apply_opencv_test_environment(obs2_postprocess_tests)

    add_executable(obs2_tracker_state_tests
        tests/cpp/tracker_state_tests.cpp
        mouse/BoxTarget.cpp)
    target_compile_definitions(obs2_tracker_state_tests
        PRIVATE
            NOMINMAX)
    target_include_directories(obs2_tracker_state_tests
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/shims"
            "${CMAKE_CURRENT_SOURCE_DIR}/mouse"
            "${CMAKE_CURRENT_SOURCE_DIR}/config"
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(obs2_tracker_state_tests
        PRIVATE
            obs2_targeting_core
            OBS2::OpenCV)
    obs2_copy_opencv_runtime(obs2_tracker_state_tests)
    add_test(NAME obs2_tracker_state_tests COMMAND obs2_tracker_state_tests)
    obs2_apply_opencv_test_environment(obs2_tracker_state_tests)
endif()
