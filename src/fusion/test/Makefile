# Makefile for thermal_to_ir_fusion test

# Cross Compiler configuration (similar to ai_ex/Makefile)
CROSS_COMPILE_BIN_PATH = /home/mark/arm-fhva12c-linux-uclibcgnueabihf-b5.3/arm-fhva12c-linux-uclibcgnueabihf-b5/bin/
CROSS_COMPILE = $(CROSS_COMPILE_BIN_PATH)arm-fhva12c-linux-uclibcgnueabihf-
CC=$(CROSS_COMPILE)gcc
CXX=$(CROSS_COMPILE)g++
AR=$(CROSS_COMPILE)ar

# Flags
CFLAGS := -Wall -O3
# CPU flags (Important for NEON)
CFLAGS += -mcpu=cortex-a7 -mfloat-abi=hard -mfpu=neon-vfpv4 
CFLAGS += -ffunction-sections -mno-unaligned-access -fno-aggressive-loop-optimizations -fstack-protector -fPIC
CFLAGS += -std=c++11

# No library links needed for this standalone test (opencv removed, stb removed)
# Standard libs
LFLAGS := -lpthread -lm -ldl -lstdc++


# Targets




TARGET_FUSION = fusion_test
TARGET_FUSION_V3 = fusion_v3
TARGET_FUSION_V3_NEON = fusion_v3_neon
TARGET_FUSION_V4 = fusion_v4
TARGET_FUSION_V5 = fusion_v5
TARGET_FUSION_V5_NEON = fusion_v5_neon
TARGET_FUSION_V6 = fusion_v6
TARGET_FUSION_V7 = fusion_v7
TARGET_FUSION_V5_1 = fusion_v5_1
TARGET_FUSION_V5_2_NEON = fusion_v5_2_neon
TARGET_FUSION_V8_1 = fusion_v8_1
TARGET_FUSION_V8_2 = fusion_v8_2
TARGET_FUSION_V8_3 = fusion_v8_3
TARGET_FUSION_V8_4 = fusion_v8_4
TARGET_POINT  = point_transform_demo
TARGET_SDK    = fusion_sdk_test

# Include Paths (Adjust as needed to find HermesII_sdk.h and image_fusion.h)
# Assuming Makefile is in src/fusion/test
# ../../.. for root, then include
INCLUDES = -I../../../include -I..

# Source for SDK Logic (We are linking statically or compiling source directly for test)
# Including HermesII_sdk.cpp and image_fusion.cpp directly to avoid linking lib issues for this standalone test
SDK_SRCS = ../../../src/HermesII_sdk.cpp ../../../src/fusion/image_fusion.cpp ../../../src/ai/model_runner.cpp ../../../src/fall/fall_detector.cpp ../../../src/ai/face_detector.cpp

all: $(TARGET_FUSION) $(TARGET_POINT) $(TARGET_FUSION_V3) $(TARGET_FUSION_V3_NEON) $(TARGET_FUSION_V4) $(TARGET_FUSION_V5) $(TARGET_FUSION_V5_NEON) $(TARGET_FUSION_V6) $(TARGET_FUSION_V7) $(TARGET_FUSION_V5_1) $(TARGET_FUSION_V5_2_NEON) $(TARGET_FUSION_V8_1) $(TARGET_FUSION_V8_2) $(TARGET_FUSION_V8_3) $(TARGET_FUSION_V8_4)

$(TARGET_FUSION): thermal_to_ir_fusion.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V3): thermal_to_ir_fusion_v3.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V3_NEON): thermal_to_ir_fusion_v3_neon.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V4): thermal_to_ir_fusion_v4.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V5): thermal_to_ir_fusion_v5.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V5_NEON): thermal_to_ir_fusion_v5_neon.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V6): thermal_to_ir_fusion_v6.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V7): thermal_to_ir_fusion_v7.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V5_1): thermal_to_ir_fusion_v5_1.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V5_2_NEON): thermal_to_ir_fusion_v5_2_neon.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V8_1): thermal_to_ir_fusion_v8_1.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V8_2): thermal_to_ir_fusion_v8_2.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V8_3): thermal_to_ir_fusion_v8_3.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_FUSION_V8_4): thermal_to_ir_fusion_v8_4.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_POINT): point_transform_demo.o
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

$(TARGET_SDK): fusion_sdk_test.cpp
	$(CXX) -o $@ $^ $(SDK_SRCS) $(CFLAGS) $(LFLAGS) $(INCLUDES)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(INCLUDES) -o $@ $<


%.o: %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET_FUSION) $(TARGET_POINT) *.o *.bmp *.jpg
