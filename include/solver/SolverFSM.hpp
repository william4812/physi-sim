#pragma once
#include <atomic> 
#include <string_view> 

namespace physi_sim
{
namespace solver
{

enum class SolverState
{
    IDLE = 0,
    READY = 1,
    RUNNING = 2,
    CONVERGED = 3,
    FAILED = 4
};

constexpr std::string_view to_string(SolverState s) noexcept 
{
  
  switch(s) 
  {
      case SolverState::IDLE : return "IDLE";
      case SolverState::READY : return "READY";
      case SolverState::RUNNING : return "RUNNING";
      case SolverState::CONVERGED : return "CONVERGED";
      case SolverState::FAILED : return "FAILED";
  }
  return "UNKNOWN";
}

/**
 * @class SolverFSM
 * @brief Thread-safe Finite State Machine governing the solver lifecycle.
 *
 * @details
 * **Concurrency Contract:** The state is managed via `std::atomic<SolverState>` with
 * memory_order_acquire/release semantics. This provides lock-free, thread-safe polling
 * for monitoring threads without the latency penalty of mutex contention.
 *
 * **Defensive Programming:** Every transition method enforces strict pre-state
 * requirements via `assertState()`. Invalid state transitions trigger a
 * `std::logic_error`, ensuring the solver fails fast rather than executing
 * indeterminate physics.
 */
class SolverFSM 
{
public: 
  SolverFSM() noexcept = default;
 
  SolverFSM(const SolverFSM&)               = delete;
  SolverFSM& operator=(const SolverFSM&)    = delete;
  SolverFSM(SolverFSM&&) noexcept           = default;
  SolverFSM& operator=(SolverFSM&&) noexcept = default;

  // Events
  void prepare();
  void start();
  void finish(bool converged);
  void reset() noexcept;

  [[nodiscard]] SolverState state() const noexcept;
  [[nodiscard]] std::string_view stateName() const noexcept;
  [[nodiscard]] bool isTerminal() const noexcept;

protected:

private:
  std::atomic<SolverState> state_{SolverState::IDLE};    
  void assertState(SolverState expected, std::string_view event) const;
};

} //namespace solver
} //namespace physi_sim
