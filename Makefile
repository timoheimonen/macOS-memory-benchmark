# Makefile for macOS Memory Benchmark

# Compiler and assembler
CXX = clang++
AS = as

# Source directory
SRC_DIR = src

# Flags for the C++ compiler
# -Wall: Enable most warnings
# -O3: High optimization level
# -std=c++17: Use C++17 standard (or newer, e.g., c++20)
# -arch arm64: Target architecture for Apple Silicon
# -pthread: Link the thread library (needed for std::thread)
# -I$(SRC_DIR): Look for headers in the src directory
CXXFLAGS = -Wall -O3 -std=c++17 -arch arm64 -pthread -I$(SRC_DIR)

# Flags for the assembler
# -arch arm64: Target architecture
ASFLAGS = -arch arm64

# Linker flags (pthread is already in CXXFLAGS, but can be here too)
LDFLAGS = -pthread

# C++ Source files
# Files in the root directory
CPP_SRCS_ROOT = main.cpp
# Files in the src directory
CPP_SRCS_SRC = timer.cpp system_info.cpp memory_utils.cpp warmup.cpp benchmark_tests.cpp utils.cpp

# Add src/ prefix to source files in the src directory
CPP_SRCS_SRC_FULL = $(addprefix $(SRC_DIR)/, $(CPP_SRCS_SRC))

# All C++ source files with correct paths
ALL_CPP_SRCS = $(CPP_SRCS_ROOT) $(CPP_SRCS_SRC_FULL)

# Assembly source files (in root)
ASM_SRCS = loops.s

# Object files (derived from source files, maintaining directory structure)
# main.cpp -> main.o
# src/timer.cpp -> src/timer.o
# loops.s -> loops.o
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

# Rule for assembling assembly source files (in root) into object files
# This handles loops.s -> loops.o
%.o: %.s
	@echo "Assembling $< -> $@..."
	$(AS) $(ASFLAGS) $< -o $@

# Clean target: remove object files (from root and src/) and the executable
clean:
	@echo "Cleaning up object files and target..."
	rm -f $(TARGET) $(OBJ_FILES)
	@echo "Cleanup complete."

# Define targets that don't correspond to files
.PHONY: all clean