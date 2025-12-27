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
# Files in the src directory (now organized in subdirectories)
CPP_SRCS_CORE_CONFIG = core/config/config.cpp
CPP_SRCS_CORE_MEMORY = core/memory/buffer_manager.cpp core/memory/memory_manager.cpp core/memory/memory_utils.cpp
CPP_SRCS_CORE_SYSTEM = core/system/system_info.cpp
CPP_SRCS_CORE_TIMING = core/timing/timer.cpp
CPP_SRCS_OUTPUT_CONSOLE = output/console/output_printer.cpp output/console/statistics.cpp
CPP_SRCS_OUTPUT_CONSOLE_MESSAGES = output/console/messages/error_messages.cpp output/console/messages/warning_messages.cpp output/console/messages/info_messages.cpp output/console/messages/program_messages.cpp output/console/messages/config_messages.cpp output/console/messages/cache_messages.cpp output/console/messages/results_messages.cpp output/console/messages/statistics_messages.cpp output/console/messages/pattern_messages.cpp
CPP_SRCS_UTILS = utils/utils.cpp utils/json_utils.cpp
CPP_SRCS_SRC = $(CPP_SRCS_CORE_CONFIG) $(CPP_SRCS_CORE_MEMORY) $(CPP_SRCS_CORE_SYSTEM) $(CPP_SRCS_CORE_TIMING) $(CPP_SRCS_OUTPUT_CONSOLE) $(CPP_SRCS_OUTPUT_CONSOLE_MESSAGES) $(CPP_SRCS_UTILS)
# Files in the src/benchmark subdirectory
CPP_SRCS_BENCHMARK = bandwidth_tests.cpp latency_tests.cpp benchmark_runner.cpp benchmark_executor.cpp benchmark_results.cpp
# Files in the src/warmup subdirectory
CPP_SRCS_WARMUP = basic_warmup.cpp latency_warmup.cpp cache_warmup.cpp pattern_warmup.cpp
# Files in the src/output/json/json_output subdirectory
CPP_SRCS_JSON_OUTPUT = output/json/json_output/builder.cpp output/json/json_output/main_memory.cpp output/json/json_output/cache.cpp output/json/json_output/patterns.cpp output/json/json_output/file_writer.cpp output/json/json_output/json_output.cpp
# Files in the src/pattern_benchmark subdirectory
CPP_SRCS_PATTERN_BENCHMARK = helpers.cpp validation.cpp execution.cpp execution_utils.cpp execution_strided.cpp execution_patterns.cpp output.cpp

# Add src/ prefix to source files
CPP_SRCS_SRC_FULL = $(addprefix $(SRC_DIR)/, $(CPP_SRCS_SRC))
# Add src/benchmark/ prefix to source files in the benchmark subdirectory
CPP_SRCS_BENCHMARK_FULL = $(addprefix $(SRC_DIR)/benchmark/, $(CPP_SRCS_BENCHMARK))
# Add src/ prefix to source files in the json_output subdirectory (already includes path)
CPP_SRCS_JSON_OUTPUT_FULL = $(addprefix $(SRC_DIR)/, $(CPP_SRCS_JSON_OUTPUT))
# Add src/warmup/ prefix to source files in the warmup subdirectory
CPP_SRCS_WARMUP_FULL = $(addprefix $(SRC_DIR)/warmup/, $(CPP_SRCS_WARMUP))
# Add src/pattern_benchmark/ prefix to source files in the pattern_benchmark subdirectory
CPP_SRCS_PATTERN_BENCHMARK_FULL = $(addprefix $(SRC_DIR)/pattern_benchmark/, $(CPP_SRCS_PATTERN_BENCHMARK))

# All C++ source files with correct paths
ALL_CPP_SRCS = $(CPP_SRCS_ROOT) $(CPP_SRCS_SRC_FULL) $(CPP_SRCS_BENCHMARK_FULL) $(CPP_SRCS_JSON_OUTPUT_FULL) $(CPP_SRCS_WARMUP_FULL) $(CPP_SRCS_PATTERN_BENCHMARK_FULL)

# Assembly source files (in src/asm)
ASM_SRCS = src/asm/memory_copy.s src/asm/memory_read.s src/asm/memory_write.s src/asm/memory_latency.s \
           src/asm/memory_read_reverse.s src/asm/memory_write_reverse.s src/asm/memory_copy_reverse.s \
           src/asm/memory_read_strided.s src/asm/memory_write_strided.s src/asm/memory_copy_strided.s \
           src/asm/memory_read_random.s src/asm/memory_write_random.s src/asm/memory_copy_random.s

# Object files (derived from source files, maintaining directory structure)
# main.cpp -> main.o
# src/timer.cpp -> src/timer.o
# memory_copy.s / memory_read.s / memory_write.s / memory_latency.s -> *.o
OBJ_FILES = $(ALL_CPP_SRCS:.cpp=.o) $(ASM_SRCS:.s=.o)

# Target executable name
TARGET = memory_benchmark

# Header file(s) that C++ files depend on (now in src/utils/)
# If benchmark.h changes, recompile C++ files that include it
HEADERS = $(SRC_DIR)/utils/benchmark.h

# Default target: build the executable
all:
	@echo "Starting build..." > /dev/null; \
	date +%s > .build_start_time; \
	$(MAKE) $(TARGET) || (rm -f .build_start_time; exit 1); \
	START_TIME=$$(cat .build_start_time); \
	rm -f .build_start_time; \
	END_TIME=$$(date +%s); \
	ELAPSED=$$((END_TIME - START_TIME)); \
	MINUTES=$$((ELAPSED / 60)); \
	SECONDS=$$((ELAPSED % 60)); \
	if [ $$MINUTES -gt 0 ]; then \
		echo "Build completed in $$MINUTES minute(s) and $$SECONDS second(s)."; \
	else \
		echo "Build completed in $$SECONDS second(s)."; \
	fi

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

# Rule for compiling C++ source files in the src/benchmark/ directory into object files
# This rule handles src/benchmark/benchmark_runner.cpp -> src/benchmark/benchmark_runner.o etc.
$(SRC_DIR)/benchmark/%.o: $(SRC_DIR)/benchmark/%.cpp $(HEADERS)
	@echo "Compiling (benchmark) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/output/json/json_output/ directory into object files
# This rule handles src/output/json/json_output/builder.cpp -> src/output/json/json_output/builder.o etc.
$(SRC_DIR)/output/json/json_output/%.o: $(SRC_DIR)/output/json/json_output/%.cpp $(HEADERS)
	@echo "Compiling (json_output) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/warmup/ directory into object files
# This rule handles src/warmup/basic_warmup.cpp -> src/warmup/basic_warmup.o etc.
$(SRC_DIR)/warmup/%.o: $(SRC_DIR)/warmup/%.cpp $(HEADERS)
	@echo "Compiling (warmup) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/pattern_benchmark/ directory into object files
# This rule handles src/pattern_benchmark/helpers.cpp -> src/pattern_benchmark/helpers.o etc.
$(SRC_DIR)/pattern_benchmark/%.o: $(SRC_DIR)/pattern_benchmark/%.cpp $(HEADERS)
	@echo "Compiling (pattern_benchmark) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/output/console/messages/ directory into object files
# This rule handles src/output/console/messages/error_messages.cpp -> src/output/console/messages/error_messages.o etc.
$(SRC_DIR)/output/console/messages/%.o: $(SRC_DIR)/output/console/messages/%.cpp $(HEADERS)
	@echo "Compiling (messages) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/core/ directories into object files
$(SRC_DIR)/core/config/%.o: $(SRC_DIR)/core/config/%.cpp $(HEADERS)
	@echo "Compiling (core/config) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/core/memory/%.o: $(SRC_DIR)/core/memory/%.cpp $(HEADERS)
	@echo "Compiling (core/memory) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/core/system/%.o: $(SRC_DIR)/core/system/%.cpp $(HEADERS)
	@echo "Compiling (core/system) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/core/timing/%.o: $(SRC_DIR)/core/timing/%.cpp $(HEADERS)
	@echo "Compiling (core/timing) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/output/console/ directory into object files
$(SRC_DIR)/output/console/%.o: $(SRC_DIR)/output/console/%.cpp $(HEADERS)
	@echo "Compiling (output/console) $< -> $@..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for compiling C++ source files in the src/utils/ directory into object files
$(SRC_DIR)/utils/%.o: $(SRC_DIR)/utils/%.cpp $(HEADERS)
	@echo "Compiling (utils) $< -> $@..."
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
	rm -f $(TARGET) $(OBJ_FILES) .build_start_time
	@echo "Cleanup complete."

# Documentation directory
DOCS_DIR = docs

# Doxygen configuration file
DOXYFILE = Doxyfile

# Generate Doxygen configuration file
$(DOXYFILE):
	@echo "Generating $(DOXYFILE)..."
	doxygen -g $(DOXYFILE)
	@echo "Configuring $(DOXYFILE)..."
	@sed -i '' 's|^PROJECT_NAME[[:space:]]*=.*|PROJECT_NAME = "macOS-memory-benchmark"|' $(DOXYFILE)
	@sed -i '' 's|^PROJECT_BRIEF[[:space:]]*=.*|PROJECT_BRIEF = "Apple silicon memory benchmark console-tool."|' $(DOXYFILE)
	@sed -i '' 's|^INPUT[[:space:]]*=.*|INPUT = $(SRC_DIR)|' $(DOXYFILE)
	@sed -i '' 's|^RECURSIVE[[:space:]]*=.*|RECURSIVE = YES|' $(DOXYFILE)
	@sed -i '' 's|^EXTRACT_ALL[[:space:]]*=.*|EXTRACT_ALL = YES|' $(DOXYFILE)
	@sed -i '' 's|^GENERATE_HTML[[:space:]]*=.*|GENERATE_HTML = YES|' $(DOXYFILE)
	@sed -i '' 's|^GENERATE_LATEX[[:space:]]*=.*|GENERATE_LATEX = NO|' $(DOXYFILE)
	@sed -i '' 's|^OUTPUT_DIRECTORY[[:space:]]*=.*|OUTPUT_DIRECTORY = $(DOCS_DIR)|' $(DOXYFILE)
	@echo "$(DOXYFILE) generated and configured."

# Generate documentation
docs: $(DOXYFILE)
	@echo "Generating documentation..."
	doxygen $(DOXYFILE)
	@echo "Documentation generated in $(DOCS_DIR)/html/"

# Clean documentation
clean-docs:
	@echo "Cleaning documentation..."
	rm -rf $(DOCS_DIR) $(DOXYFILE)
	@echo "Documentation cleanup complete."

# Define targets that don't correspond to files
.PHONY: all clean test clean-test docs clean-docs