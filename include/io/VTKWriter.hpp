#pragma once

#include "io/IResultWriter.hpp"

namespace physi_sim::io
{

class VTKWriter : public IResultWriter
{
public:
  void write(const std::vector<double>& field,
             const std::string& filename) override;

}; // class VTKWriter

} // namespace physi_sim::io
