#pragma once

#include "thermal/IThermalSolver.hpp"
#include "MockBackend.hpp"

class LBMSolver : public IThermalSolver
{
public:
    LBMSolver(std::unique_ptr<MockBackend> backend) : m_backend(std::move(backend)) 
    {
    }
    
    void step(double dt, double dx) 
    {
       m_backend->collide();
       m_backend->stream();
    }

    std::vector<double> getTemperatureField() const 
    {
        std::vector<double> dVector{0.0};
        return dVector;
    }

    void getResult() const
    {
    }

protected:

private:
  std::unique_ptr<MockBackend> m_backend;
 
};

