cmake_minimum_required(VERSION 3.20)
project(taver LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- ONNX Runtime (prebuilt, vendored in third_party) ----------------------
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
if(NOT EXISTS ${ORT_ROOT}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "ONNX Runtime not found at ${ORT_ROOT}. Run scripts/setup.ps1 to fetch it.")
endif()
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_ROOT}/lib/onnxruntime.dll
  IMPORTED_IMPLIB   ${ORT_ROOT}/lib/onnxruntime.lib
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include)

add_compile_definitions(_USE_MATH_DEFINES)  # M_PI on MSVC
if(MSVC)
  add_compile_options(/utf-8 /W3 /EHsc /O2)
endif()

# ---- Taver shared library --------------------------------------------------
add_library(taver SHARED
  src/audio/resampler.cpp
  src/audio/wav_file_source.cpp
  src/audio/wasapi_source.cpp
  src/dsp/fft.cpp
  src/dsp/feature_extractor.cpp
  src/ai/onnx_inference_engine.cpp
  src/align/vocabulary.cpp
  src/align/gop_scorer.cpp
  src/align/calibration.cpp
  src/g2p/g2p.cpp
  src/g2p/lexicon.cpp
  src/prosody/prosody.cpp
  src/pipeline/engine.cpp
  src/api/taver_api.cpp)

target_include_directories(taver
  PUBLIC  ${CMAKE_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(taver PRIVATE TAVER_BUILD_DLL)
target_link_libraries(taver PRIVATE onnxruntime)
if(WIN32)
  target_link_libraries(taver PRIVATE ole32 avrt psapi)
endif()

add_custom_command(TARGET taver POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ORT_ROOT}/lib/onnxruntime.dll $<TARGET_FILE_DIR:taver>)
if(EXISTS ${ORT_ROOT}/lib/onnxruntime_providers_shared.dll)
  add_custom_command(TARGET taver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_ROOT}/lib/onnxruntime_providers_shared.dll $<TARGET_FILE_DIR:taver>)
endif()

# ---- Console demo ----------------------------------------------------------
add_executable(taver_cli app/main.cpp)
target_include_directories(taver_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(taver_cli PRIVATE TAVER_USE_DLL)
target_link_libraries(taver_cli PRIVATE taver)

# ---- Local web server (browser UI host) ------------------------------------
add_executable(taver_server app/server.cpp)
target_include_directories(taver_server PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(taver_server PRIVATE TAVER_USE_DLL)
target_link_libraries(taver_server PRIVATE taver)

# ---- Tests -----------------------------------------------------------------
enable_testing()
add_executable(test_core tests/test_core.cpp)
target_include_directories(test_core PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME core COMMAND test_core)

add_executable(test_dsp tests/test_dsp.cpp src/dsp/fft.cpp src/dsp/feature_extractor.cpp)
target_include_directories(test_dsp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME dsp COMMAND test_dsp)

add_executable(test_g2p tests/test_g2p.cpp src/g2p/g2p.cpp src/g2p/lexicon.cpp src/align/vocabulary.cpp)
target_include_directories(test_g2p PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
add_test(NAME g2p COMMAND test_g2p WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
