#pragma once
#include <string>
#include "core/SimulationParams.hpp"

namespace physi_sim::io 
{
    
class ConfigLoader 
{
public:
    static core::SimulationParams load_json(const std::string& path);
};

} // namespace physi_sim::io 
