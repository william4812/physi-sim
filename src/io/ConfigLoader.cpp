#include "io/ConfigLoader.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace physi_sim::io 
{

core::SimulationParams ConfigLoader::load_json(const std::string& path) 
{
    // 1. Resource Acquisition: Open the file stream
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("File not found: " + path);
    }

    // 2. Parsing: Use the library to turn the file into a JSON object
    nlohmann::json data = nlohmann::json::parse(file);

    
    // 3. Mapping: Transfer data to your internal C++ 'Contract'
    core::SimulationParams params;
    params.initial_temp = data.at("thermal").at("initial_temp").get<double>();
    params.max_iterations = data.at("solver").at("max_iter").get<int>();
    params.solver_type = data.at("solver").at("type").get<std::string>();

    return params;
}

} // namespace physi_sim::io
