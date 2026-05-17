#include "solver/SolverFSM.hpp"
#include <sstream>

namespace physi_sim
{
namespace solver
{

void SolverFSM::prepare()
{
    assertState(SolverState::IDLE, "prepare");
    state_.store(SolverState::READY, std::memory_order_release);
}

void SolverFSM::start()
{
    assertState(SolverState::READY, "start");
    state_.store(SolverState::RUNNING, std::memory_order_release);
}

void SolverFSM::finish(bool converged)
{
    assertState(SolverState::RUNNING, "finish");
    state_.store(
        converged ? SolverState::CONVERGED : SolverState::FAILED,
        std::memory_order_release
    );
}

void SolverFSM::reset() noexcept 
{
    state_.store(SolverState::IDLE, std::memory_order_release);
}


SolverState SolverFSM::state() const noexcept 
{
    return state_.load(std::memory_order_acquire);
}

[[nodiscard]] std::string_view SolverFSM::stateName() const noexcept
{
    return to_string(state_);
}

[[nodiscard]] bool SolverFSM::isTerminal() const noexcept
{
    const auto s = state(); 
    return (s == SolverState::CONVERGED || s == SolverState::FAILED);
}

void SolverFSM::assertState(SolverState expected, std::string_view event) const 
{
    const auto current = state();
    if (current != expected) 
    {
        std::ostringstream oss;
        oss << "SolverFSM::" << event << "() illegal in state ["
            << to_string(current) << "] -- expected ["
            << to_string(expected) << "]";
        throw std::logic_error(oss.str());
    }
}

} //namespace solver
} //namespace physi_sim
