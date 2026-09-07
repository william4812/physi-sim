#!/bin/bash

# Automatically navigate to the repository root (one level up from scripts/)
cd "$(dirname "$0")/.."

for preset in llvm-release llvm-debug gnc-release gnc-debug; do
    echo "=================================================="
    echo "Processing Preset: $preset"
    echo "=================================================="

    cmake --preset "$preset" && \
    cmake --build --preset "$preset" -j$(nproc) && \
    ctest --preset "$preset" --output-on-failure || { echo "❌ Failure in preset: $preset"; exit 1; }

    echo "✅ Successfully completed: $preset"
    echo ""
done 2>&1 | tee Shell_Loop_Script_Result.txt
