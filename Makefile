# Makefile for macOS Memory Benchmark

# Compiler and assembler
CXX = clang++
AS = as

# Flags for the C++ compiler
# -Wall: Enable most warnings
# -O3: High optimization level
# -std=c++17: Use C++17 standard (or newer, e.g., c++20)
# -arch arm64: Target architecture for Apple Silicon
# -pthread: Link the thread library (needed for std::thread)
# -I.: Look for headers in the current directory too (just in case)
CXXFLAGS = -Wall -O3 -std=c++17 -arch arm64 -pthread -I.

# Flags for the assembler
# -arch arm64: Target architecture
ASFLAGS = -arch arm64

# Linker flags (pthread is already in CXXFLAGS, but can be here too)
LDFLAGS = -pthread

# Source files
CPP_SRCS = main.cpp timer.cpp system_info.cpp memory_utils.cpp warmup.cpp benchmark_tests.cpp utils.cpp
ASM_SRCS = loops.s

# Object files (derived from source files: .cpp -> .o, .s -> .o)
OBJ_FILES = $(CPP_SRCS:.cpp=.o) $(ASM_SRCS:.s=.o)

# Target executable name
TARGET = memory_benchmark

# Header file(s) that C++ files depend on
# If benchmark.h changes, recompile C++ files
HEADERS = benchmark.h

# Default target: build the executable
all: $(TARGET)

# Rule for linking the executable from object files
$(TARGET): $(OBJ_FILES)
	@echo "Linking $(TARGET)..."
	$(CXX) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)
	@echo "$(TARGET) built successfully."

# Rule for compiling C++ source files into object files
# $< means the first prerequisite (source file)
# $@ means the target (object file)
%.o: %.cpp $(HEADERS)
	@echo "Compiling $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for assembling assembly source files into object files
%.o: %.s
	@echo "Assembling $< -> $@..."
	$(AS) $(ASFLAGS) $< -o $@

# Clean target: remove object files and the executable
clean:
	@echo "Cleaning up object files and target..."
	rm -f $(TARGET) $(OBJ_FILES)
	@echo "Cleanup complete."

# Define targets that don't correspond to files
.PHONY: all clean