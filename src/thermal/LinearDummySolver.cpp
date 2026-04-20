#include "thermal/LinearDummySolver.hpp"
#include <iostream>

LinearDummySolver::LinearDummySolver(
        std::unique_ptr<MockBackend> backend,
        const size_t size) 
    : m_backend(std::move(backend)),
      m_temperature(size, 0.0), // 100 elements, all 0.0
      m_cnt(0)
{
}

void LinearDummySolver::step(double dt, double dx) 
{
    if (m_cnt >= MAX_CNT) return;
    m_temperature[m_cnt] = 1 + static_cast<double>(m_cnt);
    ++m_cnt;
}

std::vector<double> LinearDummySolver::getTemperatureField() const
{
    //std::vector<double> dVector{0.0};
    //return dVector;
    return m_temperature;
}

void LinearDummySolver::getResult() const
{
}
