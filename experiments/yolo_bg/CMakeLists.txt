cmake_minimum_required(VERSION 3.20)
project(person_bg_eval LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
option(HERMES_ENABLE_ONNX_PERSON "Compile Windows person backend" ON)
get_filename_component(SDK "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(WORKSPACE "${SDK}/.." ABSOLUTE)
add_executable(fall_eval_runner runner.cpp event_recorder_disabled.cpp
 ${SDK}/src/HermesII_sdk.cpp ${SDK}/src/fall/fall_detector.cpp
 ${SDK}/src/ai/model_runner.cpp ${SDK}/src/ai/face_detector.cpp
 ${SDK}/src/ai/blaze_face_detect_parser.cpp ${SDK}/src/fusion/image_fusion.cpp
 ${SDK}/src/image_process/image_process.cpp)
target_include_directories(fall_eval_runner PRIVATE ${SDK}/include ${SDK}/src)
target_compile_definitions(fall_eval_runner PRIVATE DISABLE_NPU EVENT_RECORDER_PC_ANALYSIS=0 HERMES_BG_STABILITY=1)
if(HERMES_ENABLE_ONNX_PERSON)
 target_sources(fall_eval_runner PRIVATE ${SDK}/src/ai/person_detector_onnx.cpp ${SDK}/src/fall/person_bg_controller.cpp)
 target_include_directories(fall_eval_runner PRIVATE ${WORKSPACE}/third_party/onnxruntime/include)
 target_compile_definitions(fall_eval_runner PRIVATE HERMES_ENABLE_ONNX_PERSON=1)
endif()
target_compile_options(fall_eval_runner PRIVATE -O3)
target_link_options(fall_eval_runner PRIVATE -static -static-libgcc -static-libstdc++)

if(HERMES_ENABLE_ONNX_PERSON)
 add_executable(person_probe probe.cpp ${SDK}/src/ai/person_detector_onnx.cpp ${SDK}/src/fall/person_bg_controller.cpp)
 target_include_directories(person_probe PRIVATE ${SDK}/include ${SDK}/src/ai ${SDK}/src/fall ${WORKSPACE}/third_party/onnxruntime/include)
 target_compile_options(person_probe PRIVATE -O3)
 target_link_options(person_probe PRIVATE -static -static-libgcc -static-libstdc++)
endif()

add_executable(history_two_test history_two_test.cpp)
target_link_options(history_two_test PRIVATE -static -static-libgcc -static-libstdc++)
if(HERMES_ENABLE_ONNX_PERSON)
 add_executable(model_parity_probe model_parity_probe.cpp ${SDK}/src/ai/person_detector_onnx.cpp)
 target_include_directories(model_parity_probe PRIVATE ${SDK}/src/ai ${WORKSPACE}/third_party/onnxruntime/include)
 target_compile_options(model_parity_probe PRIVATE -O3)
 target_link_options(model_parity_probe PRIVATE -static -static-libgcc -static-libstdc++)
endif()

add_executable(identity_a_test identity_a_test.cpp)
target_link_options(identity_a_test PRIVATE -static -static-libgcc -static-libstdc++)
