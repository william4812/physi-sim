#include "io/ConfigLoader.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

namespace physi_sim::io 
{

core::SimulationParams ConfigLoader::load_json(const std::string& path) 
{
    // 1. Resource Acquisition: Open the file stream
    std::ifstream file(path);
    if (!file.is_open()) 
    {
        throw std::runtime_error("File not found: " + path);
    }

    // 2. Parsing: Use the library to turn the file into a JSON object
    nlohmann::json data = nlohmann::json::parse(file);

    
    // 3. Mapping: Transfer data to your internal C++ 'Contract'
    core::SimulationParams params;
    
    params.initial_temp = data.at("thermal").at("initial_temp").get<double>();
    params.max_iterations = data.at("solver").at("max_iter").get<int>();
    params.solver_type = data.at("solver").at("type").get<std::string>();
   
    // benchmark section — optional, safe defaults apply if absent
    if (data.contains("benchmark")) 
    {
        const auto& b = data["benchmark"];
        if (b.contains("grid_sizes"))
            params.grid_sizes = b["grid_sizes"].get<std::vector<int>>();
        if (b.contains("tolerance"))
            params.tolerance  = b["tolerance"].get<double>();
        if (b.contains("run_gpu"))
            params.run_gpu    = b["run_gpu"].get<bool>();
        if (b.contains("output_dir"))
            params.output_dir = b["output_dir"].get<std::string>();
    }

    return params;
}

} // namespace physi_sim::io
