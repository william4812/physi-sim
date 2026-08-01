// tests/verification/test_anisotropic_domain.cpp
//
// Verifies the anisotropic (nx,ny,nz)+(Lx,Ly,Lz) constructor. The cubic ladder
// covers dx==dy==dz; this covers dx != dy != dz, where a per-axis area/spacing
// bug would hide.
//
// TOLERANCE NOTE: SOR drives the max cell UPDATE below `tol`; in double precision
// the achievable floor is a few 1e-13, so requesting tol=1e-13 can leave the
// solver reporting "not converged" while the SOLUTION is already correct to
// round-off. These tests therefore solve to 1e-11 (comfortably reachable) and
// assert accuracy of the FIELD, not the convergence flag.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "thermal3d/ElectroThermal3D.hpp"

using namespace physi_sim::thermal3d;
namespace {
using BCs = ElectroThermal3D::BCs; using FB = ElectroThermal3D::FaceBC;
BCs adiabatic(){ BCs b; for(auto&f:b){f.type=FB::Neumann;f.value=[](double,double,double){return 0.0;};} return b; }
void setD(BCs&b,std::size_t f,double v){ b[f].type=FB::Dirichlet; b[f].value=[v](double,double,double){return v;}; }
std::size_t ID(std::size_t ny,std::size_t nz,std::size_t i,std::size_t j,std::size_t k){ return (i*ny+j)*nz+k; }
constexpr double SOLVE_TOL = 1e-11;   // reachable SOR floor in double precision
}

// A linear field is exact on ANY grid. Non-cubic 16x8x4 with three different
// spacings: correct per-axis dx/dy/dz => Linf ~ round-off. Assert on the FIELD.
TEST(AnisotropicDomain, LinearProfileExactOnNonCubicGrid) {
    const std::size_t nx=16,ny=8,nz=4;
    const double Lx=2000e-6,Ly=1000e-6,Lz=500e-6, Th=400.0,Tc=300.0;
    ElectroThermal3D s(nx,ny,nz,Lx,Ly,Lz,[](double,double,double){return 1.0;});
    BCs b=adiabatic(); setD(b,ElectroThermal3D::ZLow,Th); setD(b,ElectroThermal3D::ZHigh,Tc);
    std::vector<double> src(nx*ny*nz,0.0), T(nx*ny*nz,0.0);
    s.solveThermal([](double,double,double){return 150.0;}, b, src, T, SOLVE_TOL);
    const double dz=Lz/nz; double worst=0.0;
    for(std::size_t i=0;i<nx;++i)for(std::size_t j=0;j<ny;++j)for(std::size_t k=0;k<nz;++k){
        double z=(k+0.5)*dz, ex=Th-(Th-Tc)*z/Lz;
        worst=std::max(worst,std::abs(T[ID(ny,nz,i,j,k)]-ex));
    }
    EXPECT_LT(worst,1e-8) << "non-cubic linear profile Linf=" << worst << " K";
}

// Direction independence: the same 1D linear problem posed along x, then y, then
// z must give the same field. Compares the CELL-CENTRE MAX across axes -- which
// is Th - (Th-Tc)*(0.5/n) for an n-cell axis, NOT Th (a centre never sits on the
// boundary). Using the same n on the driven axis makes the three directly
// comparable. Catches an axis whose area/spacing is transposed.
TEST(AnisotropicDomain, DirectionIndependenceOfLinearGradient) {
    const double Th=400.0,Tc=300.0;
    const std::size_t nd=12;   // driven-axis cells: SAME for all three so peaks match
    auto peakAlong=[&](int axis)->double{
        std::size_t nx=(axis==0?nd:6), ny=(axis==1?nd:6), nz=(axis==2?nd:6);
        ElectroThermal3D s(nx,ny,nz,1e-3,1e-3,1e-3,[](double,double,double){return 1.0;});
        BCs b=adiabatic();
        std::size_t lo=(axis==0?ElectroThermal3D::XLow:axis==1?ElectroThermal3D::YLow:ElectroThermal3D::ZLow);
        std::size_t hi=(axis==0?ElectroThermal3D::XHigh:axis==1?ElectroThermal3D::YHigh:ElectroThermal3D::ZHigh);
        setD(b,lo,Th); setD(b,hi,Tc);
        std::vector<double> src(nx*ny*nz,0.0), T(nx*ny*nz,0.0);
        s.solveThermal([](double,double,double){return 150.0;}, b, src, T, SOLVE_TOL);
        double mx=-1e9; for(double t:T) mx=std::max(mx,t); return mx;
    };
    const double ax=peakAlong(0), ay=peakAlong(1), az=peakAlong(2);
    EXPECT_NEAR(ax,ay,1e-7) << "x vs y peak differ";
    EXPECT_NEAR(ay,az,1e-7) << "y vs z peak differ";
    // sanity: the peak cell centre is half a cell inside the hot face
    const double expected_peak = Th - (Th-Tc)*(0.5/nd);
    EXPECT_NEAR(ax, expected_peak, 1e-6) << "peak should be Th - (Th-Tc)*(0.5/n) = " << expected_peak;
}
