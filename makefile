# 指定交叉編譯器 (User's Toolchain)
CXX = /home/mark/arm-fhva12c-linux-uclibcgnueabihf-b5.3/arm-fhva12c-linux-uclibcgnueabihf-b5/bin/arm-fhva12c-linux-uclibcgnueabihf-g++
#CXX ?= g++

# 設定編譯參數
# 包含 ARM NEON 優化參數 (保留使用者的設定)
TARGET_ARCH = -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard

# NPU SDK Path (from ai_ex/Makefile)
SDK_INCLUDE = /home/mark/nn_release/npu_inference/mpp
SDK_LIBRARY = $(SDK_INCLUDE)/linux/va12c/lib

# 通用編譯參數 (+ -fPIC for shared library)
COMMON_FLAGS = -Wall -O3 -fPIC -DNDEBUG
COMMON_FLAGS += -Iinclude -Isrc 
COMMON_FLAGS += -I$(SDK_INCLUDE)/include -I$(SDK_INCLUDE)/drv_include

# ARM Specific Flags
ARM_FLAGS = $(TARGET_ARCH)
ARM_FLAGS += -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -fstack-protector

# PC Specific Flags (Empty for now, or -march=native)
PC_FLAGS = -D__PC_BUILD__ -DDISABLE_NPU

# Combine for default CXX (ARM)
CXXFLAGS = $(COMMON_FLAGS) $(ARM_FLAGS)

# NPU Libraries
LDFLAGS_NPU += -Wl,--start-group
LDFLAGS_NPU += $(wildcard $(SDK_LIBRARY)/npu/*.a)
LDFLAGS_NPU += $(SDK_LIBRARY)/libmpi.a
LDFLAGS_NPU += -Wl,--end-group

# Add to linker flags for targets
# Note: Since we are building a SHARED library (.so), we might need to link these in or just let the app link them?
# If FaceDetector is inside libHermesII_sdk.so, and it uses NPU symbols, we should link them if they are static libs (.a)
# However, usually static libs into shared lib requires -fPIC in the static libs.
# Assuming they are compatible.
# Wait, typically we might just link against them.


# 來源檔案
SDK_SRCS = src/HermesII_sdk.cpp \
           src/ai/model_runner.cpp \
           src/ai/face_detector.cpp \
           src/ai/blaze_face_detect_parser.cpp \
           src/fusion/image_fusion.cpp \
           src/fall/fall_detector.cpp \
           src/image_process/image_process.cpp

SDK_OBJS = $(SDK_SRCS:.cpp=.o)

# 目標檔案
BUILD_DIR = build
LIB_NAME = libHermesII_sdk.so
LIB_TARGET = $(BUILD_DIR)/$(LIB_NAME)
LIB_PC_TARGET = $(BUILD_DIR)/libHermesII_sdk_pc.so

# FALL_DEMO_TARGET = fall_demo_edge
# FALL_CALLBACK_TARGET = fall_callback_demo_edge
# DEMO_APP_TARGET = demo_app_edge
# STREAMING_DEMO_TARGET = streaming_demo_edge

FALL_DEMO_TARGET = fall_demo_edge
FALL_CALLBACK_TARGET = fall_callback_demo_edge
SDK_DEMO_CODE_V1_TARGET = SDK_DemoCode_v1_edge
SDK_DEMO_CODE_V2_TARGET = SDK_DemoCode_v2_edge
SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_TARGET = SDK_DemoCode_v2_save_fall_result_edge
SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_PC_TARGET = SDK_DemoCode_v2_save_fall_result_pc
SDK_DEMO_TEST_ACCURACY_TARGET = SDK_DemoCode_v1_test_accuracy
SDK_DEMO_BG_TEST_TARGET = SDK_DemoCode_BG_Test
ACCELERATION_TEST_TARGET = acceleration_test
DEMO_APP_TARGET = demo_app_edge
STREAMING_DEMO_TARGET = streaming_demo_edge

# 編譯 Batch Accuracy Test
FALL_ACCURACY_TEST_TARGET = fall_accuracy_test

# 預設規則
all: $(LIB_TARGET) $(LIB_PC_TARGET) $(FALL_DEMO_TARGET) $(FALL_CALLBACK_TARGET) $(SDK_DEMO_CODE_V1_TARGET) $(SDK_DEMO_CODE_V2_TARGET) $(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_TARGET) $(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_PC_TARGET) $(SDK_DEMO_TEST_ACCURACY_TARGET) $(STREAMING_DEMO_TARGET) $(FALL_ACCURACY_TEST_TARGET) $(SDK_DEMO_BG_TEST_TARGET) $(ACCELERATION_TEST_TARGET)


# 預設規則
# 預設規則
# 預設規則
all: $(LIB_TARGET) $(FALL_DEMO_TARGET) $(FALL_CALLBACK_TARGET) $(SDK_DEMO_CODE_V1_TARGET) $(SDK_DEMO_CODE_V2_TARGET) $(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_TARGET) $(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_PC_TARGET) $(SDK_DEMO_TEST_ACCURACY_TARGET) $(STREAMING_DEMO_TARGET) $(FALL_ACCURACY_TEST_TARGET) $(SDK_DEMO_BG_TEST_TARGET) $(ACCELERATION_TEST_TARGET)

# 建立 build 目錄
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# 編譯共用函式庫 (.so) - ARM
$(LIB_TARGET): $(SDK_OBJS) | $(BUILD_DIR)
	$(CXX) -shared -o $@ $(SDK_OBJS) $(LDFLAGS_NPU)

# 編譯共用函式庫 (.so) - PC (Host)
# Recompile objects for PC to avoid Arch mismatch
SDK_PC_OBJS = $(addprefix $(BUILD_DIR)/pc_, $(notdir $(SDK_OBJS)))

$(BUILD_DIR)/pc_%.o: src/%.cpp | $(BUILD_DIR)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -c $< -o $@

$(BUILD_DIR)/pc_%.o: src/ai/%.cpp | $(BUILD_DIR)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -c $< -o $@

$(BUILD_DIR)/pc_%.o: src/fusion/%.cpp | $(BUILD_DIR)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -c $< -o $@

$(BUILD_DIR)/pc_%.o: src/fall/%.cpp | $(BUILD_DIR)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -c $< -o $@

$(BUILD_DIR)/pc_%.o: src/image_process/%.cpp | $(BUILD_DIR)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -c $< -o $@

$(LIB_PC_TARGET): $(SDK_PC_OBJS) | $(BUILD_DIR)
	g++ -shared -o $@ $(SDK_PC_OBJS)

# 編譯 Fall Demo
$(FALL_DEMO_TARGET): examples/fall_demo.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)'

# 編譯 Demo App
#$(DEMO_APP_TARGET): examples/demo_app.cpp $(LIB_TARGET)
#	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' --unused

# 編譯 Demo App (Uncommenting for potentially fixing if user uncommented elsewhere, but user commented it out. I will just fix the active targets)

# 編譯 Callback Demo
$(FALL_CALLBACK_TARGET): examples/fall_callback_demo.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code v1
$(SDK_DEMO_CODE_V1_TARGET): examples/SDK_DemoCode_v1.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code v2
# 編譯 SDK Demo Code v2
$(SDK_DEMO_CODE_V2_TARGET): examples/SDK_DemoCode_v2.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code v2 Save Result
$(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_TARGET): examples/SDK_DemoCode_v2_save_fall_result.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code v2 Save Result (PC)
$(SDK_DEMO_CODE_V2_SAVE_FALL_RESULT_PC_TARGET): examples/SDK_DemoCode_v2_save_fall_result.cpp $(LIB_PC_TARGET)
	g++ $(COMMON_FLAGS) $(PC_FLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk_pc -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code v1 Accuracy Test
$(SDK_DEMO_TEST_ACCURACY_TARGET): examples/SDK_DemoCode_v1_test_accuracy.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 SDK Demo Code BG Test
$(SDK_DEMO_BG_TEST_TARGET): examples/SDK_DemoCode_BG_Test.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

# 編譯 Streaming Demo
$(STREAMING_DEMO_TARGET): examples/streaming_demo.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

$(FALL_ACCURACY_TEST_TARGET): examples/fall_callback_demo_test_accuracy.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

$(FALL_ACCURACY_TEST_TARGET): examples/fall_callback_demo_test_accuracy.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread

$(ACCELERATION_TEST_TARGET): acceleration_test.cpp $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< -L$(BUILD_DIR) -lHermesII_sdk -Wl,-rpath,'$$ORIGIN/$(BUILD_DIR)' -lpthread
# 編譯 .o 檔
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 清除規則
clean:
	rm -f $(SDK_OBJS) $(FALL_DEMO_TARGET) $(FALL_CALLBACK_TARGET) $(SDK_DEMO_CODE_V1_TARGET) $(SDK_DEMO_CODE_V2_TARGET) $(SDK_DEMO_TEST_ACCURACY_TARGET) $(STREAMING_DEMO_TARGET) $(FALL_ACCURACY_TEST_TARGET) $(SDK_DEMO_BG_TEST_TARGET)
	rm -rf $(BUILD_DIR)
