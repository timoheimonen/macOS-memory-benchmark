# Makefile for macOS Memory Benchmark

# Compiler and assembler
CXX = clang++
AS = as

# Source directory
SRC_DIR = src

# Google Test configuration
GTEST_DIR = /opt/homebrew/opt/googletest
GTEST_INCLUDE = $(GTEST_DIR)/include
GTEST_LIB_DIR = $(GTEST_DIR)/lib
GTEST_LIBS = -L$(GTEST_LIB_DIR) -lgtest -lgtest_main -pthread

# Flags for the C++ compiler
# -Wall: Enable most warnings
# -O3: High optimization level
# -std=c++17: Use C++17 standard (or newer, e.g., c++20)
# -arch arm64: Target architecture for Apple Silicon
# -pthread: Link the thread library (needed for std::thread)
# -I$(SRC_DIR): Look for headers in the src directory
CXXFLAGS = -Wall -O3 -std=c++17 -arch arm64 -pthread -I$(SRC_DIR)

# Test-specific flags (less optimization for faster compilation, debug symbols)
TEST_CXXFLAGS = -Wall -g -std=c++17 -arch arm64 -pthread -I$(SRC_DIR) -I$(GTEST_INCLUDE)

# Flags for the assembler
# -arch arm64: Target architecture
ASFLAGS = -arch arm64

# Linker flags (pthread is already in CXXFLAGS, but can be here too)
LDFLAGS = -pthread

# C++ Source files
# Files in the root directory
CPP_SRCS_ROOT = main.cpp
# Files in the src directory
CPP_SRCS_SRC = timer.cpp system_info.cpp memory_utils.cpp warmup.cpp benchmark_tests.cpp utils.cpp output_printer.cpp statistics.cpp memory_manager.cpp config.cpp buffer_manager.cpp benchmark_runner.cpp benchmark_executor.cpp benchmark_results.cpp messages.cpp

# Add src/ prefix to source files in the src directory
CPP_SRCS_SRC_FULL = $(addprefix $(SRC_DIR)/, $(CPP_SRCS_SRC))

# All C++ source files with correct paths
ALL_CPP_SRCS = $(CPP_SRCS_ROOT) $(CPP_SRCS_SRC_FULL)

# Assembly source files (in src/asm)
ASM_SRCS = src/asm/memory_copy.s src/asm/memory_read.s src/asm/memory_write.s src/asm/memory_latency.s

# Object files (derived from source files, maintaining directory structure)
# main.cpp -> main.o
# src/timer.cpp -> src/timer.o
# memory_copy.s / memory_read.s / memory_write.s / memory_latency.s -> *.o
OBJ_FILES = $(ALL_CPP_SRCS:.cpp=.o) $(ASM_SRCS:.s=.o)

# Target executable name
TARGET = memory_benchmark

# Header file(s) that C++ files depend on (now in src/)
# If benchmark.h changes, recompile C++ files that include it
HEADERS = $(SRC_DIR)/benchmark.h

# Default target: build the executable
all: $(TARGET)

# Rule for linking the executable from object files
$(TARGET): $(OBJ_FILES)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)
	@echo "$(TARGET) built successfully."

# Test directory and files
TEST_DIR = tests
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

# Test executable name
TEST_TARGET = test_runner

# Source files needed for tests (excluding main.cpp)
TEST_LIB_SRCS = $(filter-out main.cpp, $(ALL_CPP_SRCS))
TEST_LIB_OBJS = $(filter-out main.o, $(OBJ_FILES))

# Rule for compiling test files (must come before generic %.o rule)
$(TEST_DIR)/%.o: $(TEST_DIR)/%.cpp
	@echo "Compiling test $< -> $@..."
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the root directory into object files
# $< means the first prerequisite (source file)
# $@ means the target (object file)
# This rule handles main.cpp -> main.o
%.o: %.cpp $(HEADERS)
	@echo "Compiling (root) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/ directory into object files
# This rule handles src/timer.cpp -> src/timer.o etc.
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@echo "Compiling (src) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for assembling assembly source files in src/asm into object files
$(SRC_DIR)/asm/%.o: $(SRC_DIR)/asm/%.s
	@echo "Assembling $< -> $@..."
	$(AS) $(ASFLAGS) $< -o $@

# Test target: build and run tests
test: $(TEST_TARGET)
	@echo "Running tests..."
	./$(TEST_TARGET)

# Build test executable
$(TEST_TARGET): $(TEST_OBJS) $(TEST_LIB_OBJS)
	@echo "Linking $(TEST_TARGET)..."
	$(CXX) $(TEST_OBJS) $(TEST_LIB_OBJS) -o $(TEST_TARGET) $(GTEST_LIBS) $(LDFLAGS)
	@echo "$(TEST_TARGET) built successfully."

# Clean test files
clean-test:
	@echo "Cleaning test files..."
	rm -f $(TEST_TARGET) $(TEST_OBJS)
	@echo "Test cleanup complete."

# Clean target: remove object files (from root and src/) and the executable
clean: clean-test
	@echo "Cleaning up object files and target..."
	rm -f $(TARGET) $(OBJ_FILES)
	@echo "Cleanup complete."

# Define targets that don't correspond to files
.PHONY: all clean test clean-test