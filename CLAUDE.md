# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Bill's engineering standards

- Never break existing tests — run test suite before every commit
- Surgical changes only — one logical change per commit
- Every public method needs a comment stating its contract
- CMake: target_include_directories PRIVATE/PUBLIC never global include_directories
- C++17, -Wall -Wextra, no warnings allowed in CI
- Commit messages: type: description (feat/fix/build/docs/refactor/perf/test)

## Daily benchmark workflow
```bash
cmake --build --preset hpc
cd build && ./physi_sim
python3 ../python/scripts/generate_benchmark_report.py \
    --output-dir . --save-dir ../docs/figures
```

## Phase 4 — feat/phase2-vram-resident

Goal: eliminate per-iteration PCIe transfers from CudaJacobiSolver.
Current:  step() = upload + kernel + download  (PCIe every iteration)
Target:   upload() once → solve_vram() loop → download() once

Expected result at 100×100:
  Phase 1: 4,195 iters × 0.121ms/iter = 508ms
  Phase 4: 4,195 iters × 0.001ms/iter = ~4ms  (18× faster than CPU)

Key files:
  include/solver/CudaJacobiSolver.hpp  — add upload/solve_vram/download
  src/cuda/CudaJacobiSolver.cu         — implement the three new methods
  tests/unit/test_cuda_jacobi_solver.cpp — add VRAMResident tests (TDD first)
  src/main.cpp                          — ProfilingHarness on Phase 4 path
