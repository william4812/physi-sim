// tests/unit/test_field_export_vti.cpp
//
// UNIT tests for io::VTKWriter::write_3d — the SERIALISATION, not the physics
// (physics is pinned by the verification ladder). The one real hazard is INDEX
// ORDER: VTK iterates x fastest, the solver flattens k fastest. A transposed
// writer still produces a file that renders — the picture is just wrong — so
// AxisOrder writes a field varying along one axis only and checks where the
// variation lands in the serialised stream.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "io/VTKWriter.hpp"

using physi_sim::io::VTKWriter;

namespace {
std::string slurp(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}
}  // namespace

TEST(FieldExportVTI, WritesWellFormedFileWithDeclaredArrays) {
    const std::size_t n = 4; const double dx = 0.25;
    std::vector<double> a(n*n*n, 1.5), vx(n*n*n, 1.0), vy(n*n*n, 2.0), vz(n*n*n, 3.0);
    const std::string path = "vti_unit_basic.vti";

    VTKWriter w;
    ASSERT_TRUE(w.write_3d(a, "alpha", n, dx, path, &vx, &vy, &vz, "flux"));
    const std::string s = slurp(path);

    EXPECT_NE(s.find("<VTKFile type=\"ImageData\""), std::string::npos);
    EXPECT_NE(s.find("WholeExtent=\"0 4 0 4 0 4\""), std::string::npos);
    EXPECT_NE(s.find("Name=\"alpha\""), std::string::npos);
    EXPECT_NE(s.find("Name=\"flux\" NumberOfComponents=\"3\""), std::string::npos);
    EXPECT_NE(s.find("</VTKFile>"), std::string::npos);
    std::remove(path.c_str());
}

TEST(FieldExportVTI, ScalarOnlyOmitsVectorArray) {
    const std::size_t n = 3; const double dx = 1.0;
    std::vector<double> a(n*n*n, 7.0);
    const std::string path = "vti_unit_scalar.vti";

    VTKWriter w;
    ASSERT_TRUE(w.write_3d(a, "temperature_C", n, dx, path));   // no vector args
    const std::string s = slurp(path);
    EXPECT_NE(s.find("Name=\"temperature_C\""), std::string::npos);
    EXPECT_EQ(s.find("NumberOfComponents=\"3\""), std::string::npos);  // must be absent
    std::remove(path.c_str());
}

TEST(FieldExportVTI, AxisOrderMatchesVtkConvention) {
    // Field = i (solver x index). VTK writes x FASTEST, so the first n values
    // must be 0,1,2,...; if constant instead, the writer transposed x and z.
    const std::size_t n = 3; const double dx = 1.0;
    std::vector<double> f(n*n*n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                f[(i*n + j)*n + k] = static_cast<double>(i);

    const std::string path = "vti_unit_axis.vti";
    VTKWriter w;
    ASSERT_TRUE(w.write_3d(f, "xi", n, dx, path));
    const std::string s = slurp(path);

    const auto tagEnd = s.find("format=\"ascii\">");
    ASSERT_NE(tagEnd, std::string::npos);
    std::istringstream vals(s.substr(s.find('\n', tagEnd) + 1));
    double v0, v1, v2; vals >> v0 >> v1 >> v2;
    EXPECT_DOUBLE_EQ(v0, 0.0);
    EXPECT_DOUBLE_EQ(v1, 1.0);
    EXPECT_DOUBLE_EQ(v2, 2.0);
    std::remove(path.c_str());
}
