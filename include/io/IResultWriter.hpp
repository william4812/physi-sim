#pragma once

#include <vector>
#include <string>

class IResultWriter
{
public:
  virtual ~IResultWriter() = default;

  virtual void write(const std::vector<double>& field,
                     const std::string& filename) = 0; 
  virtual void write_2d(const double* field, 
                        int nx, int ny, const std::string& filename) = 0; 
};
