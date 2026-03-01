#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include "MockBackend.hpp"

class LBMSolver
{
public:
    LBMSolver(std::unique_ptr<MockBackend> backend) : m_backend(std::move(backend)) 
    {
    }
    
    void step() 
    {
       m_backend->collide();
       m_backend->stream();
    }

protected:

private:
  std::unique_ptr<MockBackend> m_backend;
 
};

#endif //LBM_SOLVER_HPP
