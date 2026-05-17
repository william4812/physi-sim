#include <gtest/gtest.h>
#include <fstream>
#include "io/VTKWriter.hpp"

using namespace physi_sim::io;

TEST(VTKWriterTest, CreatesValidHeader) {
    // 1. Setup
    VTKWriter writer;
    std::string testFile = "test_output.vtk";
    std::vector<double> dummyData = {1.0, 2.0, 3.0};

    // 2. Action
    writer.write(dummyData, testFile);

    // 3. Assert
    std::ifstream inFile(testFile);
    ASSERT_TRUE(inFile.is_open());

    std::string line;
    std::getline(inFile, line);
    EXPECT_EQ(line, "# vtk DataFile Version 3.0"); // Verify the "Contract"
    
    inFile.close();
    std::remove(testFile.c_str()); // Cleanup
}
