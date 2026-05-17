// tests/test_solver_fsm.cpp
#include <gtest/gtest.h>
#include "solver/SolverFSM.hpp"
#include <future>
#include <thread>
#include <string>

using namespace physi_sim::solver;

// ── Helper ────────────────────────────────────────────────────────────────────
static void makeRunning(SolverFSM& fsm) 
{
    fsm.prepare();
    fsm.start();
}

// ── 1. Initial state ──────────────────────────────────────────────────────────
TEST(SolverFSMTest, InitialStateIsIdle) 
{
    SolverFSM fsm;
    EXPECT_EQ(fsm.state(),      SolverState::IDLE);
    EXPECT_EQ(fsm.stateName(),  "IDLE");
    EXPECT_FALSE(fsm.isTerminal());
}

// ── 2. Valid transitions ───────────────────────────────────────────────────────
TEST(SolverFSMTest, IdleToReadyViaPrepare) 
{
    SolverFSM fsm;
    fsm.prepare();
    EXPECT_EQ(fsm.state(), SolverState::READY);
    EXPECT_EQ(fsm.stateName(),  "READY");
}


TEST(SolverFSMTest, ReadyToRunningViaStart) 
{
    SolverFSM fsm;
    fsm.prepare();
    fsm.start();
    EXPECT_EQ(fsm.state(), SolverState::RUNNING);
    EXPECT_EQ(fsm.stateName(),  "RUNNING");
}

TEST(SolverFSMTest, RunningToConvergedViaFinishTrue) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(true);
    EXPECT_EQ(fsm.state(),     SolverState::CONVERGED);
    EXPECT_EQ(fsm.stateName(), "CONVERGED");
    EXPECT_TRUE(fsm.isTerminal());
}

TEST(SolverFSMTest, RunningToFailedViaFinishFalse) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(false);
    EXPECT_EQ(fsm.state(),     SolverState::FAILED);
    EXPECT_EQ(fsm.stateName(), "FAILED");
    EXPECT_TRUE(fsm.isTerminal());
}


// ── 3. reset() from every state ───────────────────────────────────────────────
TEST(SolverFSMTest, ResetFromIdle) 
{
    SolverFSM fsm;
    fsm.reset();
    EXPECT_EQ(fsm.state(), SolverState::IDLE);
}

TEST(SolverFSMTest, ResetFromReady) 
{
    SolverFSM fsm;
    fsm.prepare();
    fsm.reset();
    EXPECT_EQ(fsm.state(), SolverState::IDLE);
}

TEST(SolverFSMTest, ResetFromRunning) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.reset();
    EXPECT_EQ(fsm.state(), SolverState::IDLE);
}

TEST(SolverFSMTest, ResetFromConvergedThenReuseFullCycle) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(true);
    fsm.reset();
    EXPECT_EQ(fsm.state(), SolverState::IDLE);
    // Verify reuse works
    fsm.prepare();
    fsm.start();
    fsm.finish(false);
    EXPECT_EQ(fsm.state(), SolverState::FAILED);
}

TEST(SolverFSMTest, ResetFromFailed) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(false);
    fsm.reset();
    EXPECT_EQ(fsm.state(), SolverState::IDLE);
}

// ── 4. Invalid transitions — must throw (9 cases) ────────────────────────────
TEST(SolverFSMTest, StartFromIdleThrows) 
{
    SolverFSM fsm;
    EXPECT_THROW(fsm.start(), std::logic_error);
}

TEST(SolverFSMTest, FinishTrueFromIdleThrows) 
{
    SolverFSM fsm;
    EXPECT_THROW(fsm.finish(true), std::logic_error);
}

TEST(SolverFSMTest, FinishFalseFromIdleThrows) 
{
    SolverFSM fsm;
    EXPECT_THROW(fsm.finish(false), std::logic_error);
}

TEST(SolverFSMTest, PrepareFromReadyThrows) 
{
    SolverFSM fsm;
    fsm.prepare();
    EXPECT_THROW(fsm.prepare(), std::logic_error);
}

TEST(SolverFSMTest, FinishFromReadyThrows) 
{
    SolverFSM fsm;
    fsm.prepare();
    EXPECT_THROW(fsm.finish(true), std::logic_error);
}

TEST(SolverFSMTest, PrepareFromRunningThrows) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    EXPECT_THROW(fsm.prepare(), std::logic_error);
}

TEST(SolverFSMTest, StartFromRunningThrows) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    EXPECT_THROW(fsm.start(), std::logic_error);
}

TEST(SolverFSMTest, PrepareFromConvergedThrows) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(true);
    EXPECT_THROW(fsm.prepare(), std::logic_error);
}

TEST(SolverFSMTest, FinishFromConvergedThrows) 
{
    SolverFSM fsm;
    makeRunning(fsm);
    fsm.finish(true);
    EXPECT_THROW(fsm.finish(true), std::logic_error);
}

// ── 5. Error message content ──────────────────────────────────────────────────
TEST(SolverFSMTest, ErrorMessageNamesCurrentStateAndEvent) 
{
    SolverFSM fsm;
    try 
    {
        fsm.start();
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) 
    {
        const std::string msg(e.what());
        EXPECT_NE(msg.find("IDLE"),  std::string::npos) << msg;
        EXPECT_NE(msg.find("start"), std::string::npos) << msg;
        EXPECT_NE(msg.find("READY"), std::string::npos) << msg;
    }
}

// ── 6. Thread-safety ──────────────────────────────────────────────────────────
TEST(SolverFSMTest, ConcurrentStateReadsAreSafe) 
{
    SolverFSM fsm;
    fsm.prepare();
    fsm.start();

    auto reader = std::async(std::launch::async, [&fsm]() 
    {
        SolverState last = SolverState::RUNNING;
        for (int i = 0; i < 10'000; ++i)
            last = fsm.state();
        return last;
    });

    std::this_thread::sleep_for(std::chrono::microseconds(50));
    fsm.finish(true);

    const auto seen = reader.get();
    EXPECT_TRUE(seen == SolverState::RUNNING || seen == SolverState::CONVERGED)
        << "Unexpected state: " << to_string(seen);
    EXPECT_EQ(fsm.state(), SolverState::CONVERGED);
}

// ── 7. Three-cycle reuse ───────────────────────────────────────────────────────
TEST(SolverFSMTest, FullCycleThreeTimes) 
{
    SolverFSM fsm;
    for (int run = 0; run < 3; ++run) 
    {
        SCOPED_TRACE("run=" + std::to_string(run));
        fsm.prepare();
        fsm.start();
        fsm.finish(run % 2 == 0);
        EXPECT_TRUE(fsm.isTerminal());
        fsm.reset();
        EXPECT_EQ(fsm.state(), SolverState::IDLE);
    }
}
