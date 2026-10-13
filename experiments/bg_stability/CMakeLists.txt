cmake_minimum_required(VERSION 3.20)
project(bg_stability_eval LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(SDK "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(fall_eval_runner runner.cpp event_recorder_disabled.cpp
 ${SDK}/src/HermesII_sdk.cpp ${SDK}/src/fall/fall_detector.cpp
 ${SDK}/src/ai/model_runner.cpp ${SDK}/src/ai/face_detector.cpp
 ${SDK}/src/ai/blaze_face_detect_parser.cpp ${SDK}/src/fusion/image_fusion.cpp
 ${SDK}/src/image_process/image_process.cpp)
target_include_directories(fall_eval_runner PRIVATE ${SDK}/include ${SDK}/src)
target_compile_definitions(fall_eval_runner PRIVATE DISABLE_NPU EVENT_RECORDER_PC_ANALYSIS=0 HERMES_BG_STABILITY=1)
target_compile_options(fall_eval_runner PRIVATE -O3)
target_link_options(fall_eval_runner PRIVATE -static -static-libgcc -static-libstdc++)
