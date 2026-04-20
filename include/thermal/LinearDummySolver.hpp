#pragma once

#include "MockBackend.hpp"
#include "thermal/IThermalSolver.hpp"
#include <cstdint>

constexpr uint64_t MAX_CNT = 100;

class LinearDummySolver : public IThermalSolver
{
public:
    //LinearDummySolver() = default;
    LinearDummySolver(std::unique_ptr<MockBackend> backend,
            const size_t size);
    void step(double dt, double dx);
    std::vector<double> getTemperatureField() const;
    void getResult() const;

protected:

private:
  std::unique_ptr<MockBackend> m_backend;
  std::vector<double> m_temperature;
  uint64_t m_cnt;
};
