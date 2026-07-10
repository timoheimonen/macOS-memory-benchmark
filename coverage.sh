#!/bin/sh
# Reproducible LLVM source coverage for production C++ without modifying the
# normal workspace build artifacts.
set -eu

MODE="${1:-}"
case "$MODE" in
  unit|all) ;;
  *)
    echo "usage: $0 unit|all" >&2
    exit 2
    ;;
esac

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="/tmp/membenchmark-coverage-$MODE"
WORK_DIR="$OUTPUT_DIR/work"

rm -rf "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT INT TERM

rsync -a \
  --exclude '.git/' \
  --exclude '*.o' \
  --exclude '*.d' \
  --exclude 'memory_benchmark' \
  --exclude 'test_runner' \
  --exclude 'docs/' \
  --exclude 'plans/' \
  "$ROOT_DIR/" "$WORK_DIR/"

COVERAGE_FLAGS='-fprofile-instr-generate -fcoverage-mapping'
if [ "$MODE" = unit ]; then
  BUILD_TARGETS="test_runner"
  COVERAGE_BINARY="$WORK_DIR/test_runner"
else
  BUILD_TARGETS="test_runner memory_benchmark"
  COVERAGE_BINARY="$WORK_DIR/memory_benchmark"
fi

# Instrument production objects only. Test translation units still link against
# the coverage runtime but cannot pollute the production report or collide with
# executable profiles through test-only inline/template functions.
make -C "$WORK_DIR" $BUILD_TARGETS \
  CXXFLAGS="-Wall -O0 -g -std=c++17 -arch arm64 -pthread -Isrc $COVERAGE_FLAGS" \
  TEST_CXXFLAGS="-Wall -O0 -g -std=c++17 -arch arm64 -pthread -Isrc "\
"-I/opt/homebrew/opt/googletest/include" \
  LDFLAGS="-pthread $COVERAGE_FLAGS"

export LLVM_PROFILE_FILE="$OUTPUT_DIR/profile-%m-%p.profraw"
if [ "$MODE" = unit ]; then
  if ! (cd "$WORK_DIR" && ./test_runner '--gtest_filter=-*Integration*') \
      > "$OUTPUT_DIR/test.log" 2>&1; then
    cat "$OUTPUT_DIR/test.log"
    exit 1
  fi
else
  if ! (cd "$WORK_DIR" && ./test_runner) \
      > "$OUTPUT_DIR/test.log" 2>&1; then
    cat "$OUTPUT_DIR/test.log"
    exit 1
  fi
fi
tail -n 20 "$OUTPUT_DIR/test.log"

xcrun llvm-profdata merge -sparse "$OUTPUT_DIR"/*.profraw \
  -o "$OUTPUT_DIR/coverage.profdata"

# llvm-cov source filters are file paths, not directory roots. Build the
# argument vector one path at a time so spaces remain quoted and only
# production C++ translation units (including main.cpp) enter the report.
SOURCE_LIST="$OUTPUT_DIR/production-sources.txt"
{
  if [ "$MODE" = all ]; then
    printf '%s\n' "$WORK_DIR/main.cpp"
  fi
  find "$WORK_DIR/src" -type f -name '*.cpp' \
    ! -path '*/third_party/*' -print
} | LC_ALL=C sort > "$SOURCE_LIST"

set -- "$COVERAGE_BINARY" \
  -instr-profile="$OUTPUT_DIR/coverage.profdata" \
  --ignore-filename-regex='/(tests|third_party)/|/opt/homebrew/opt/googletest/' \
  --sources
while IFS= read -r source_file; do
  set -- "$@" "$source_file"
done < "$SOURCE_LIST"
xcrun llvm-cov report "$@" > "$OUTPUT_DIR/report.txt"

cat "$OUTPUT_DIR/report.txt"
echo "Coverage report: $OUTPUT_DIR/report.txt"
