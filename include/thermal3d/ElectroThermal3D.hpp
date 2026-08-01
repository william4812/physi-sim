#pragma once
#include <array>
#include <cstddef>
#include <functional>
#include <vector>
namespace physi_sim::thermal3d {
class ElectroThermal3D {
public:
    using ScalarField = std::function<double(double,double,double)>;
    enum Face : std::size_t { XLow=0, XHigh, YLow, YHigh, ZLow, ZHigh, FaceCount };
    struct FaceBC {
        enum Type { Dirichlet, Neumann, Robin };
        Type type = Neumann;
        ScalarField value = [](double,double,double){ return 0.0; };
        double h = 0.0;
    };
    using BCs = std::array<FaceBC, FaceCount>;

    // ANISOTROPIC constructor: independent cell counts and lengths per axis.
    // A package is ~6 mm lateral by ~0.5 mm tall, so a cubic grid cannot
    // represent one. Every axis carries its own dx = L/n.
    ElectroThermal3D(std::size_t nx, std::size_t ny, std::size_t nz,
                     double Lx, double Ly, double Lz, ScalarField sigma);

    // CUBIC convenience constructor: forwards to the anisotropic one with
    // n,n,n and L,L,L. Every existing call site (MMS, ladder L0-L6) keeps
    // working unchanged -- the cubic path is a special case of the general one.
    ElectroThermal3D(std::size_t n, double length, ScalarField sigma)
        : ElectroThermal3D(n, n, n, length, length, length, std::move(sigma)) {}

    void setElectricalBC(const BCs& bc);
    bool solve(double tol, std::size_t maxIter);
    bool solveThermal(const ScalarField& k_field, const BCs& bcs,
                      const std::vector<double>& src, std::vector<double>& T, double tol);
    bool solveField(const std::vector<double>& gamma, const BCs& bc,
                    const std::vector<double>& src, std::vector<double>& phi,
                    double tol, std::size_t maxIter);
    double voltage(std::size_t i,std::size_t j,std::size_t k) const;
    double jouleDensity(std::size_t i,std::size_t j,std::size_t k) const;
    double totalJoulePower() const;

    // Cell-centre coordinates, now axis-aware.
    double centerX(std::size_t i) const { return (double(i)+0.5)*dx_; }
    double centerY(std::size_t j) const { return (double(j)+0.5)*dy_; }
    double centerZ(std::size_t k) const { return (double(k)+0.5)*dz_; }
    // Backward-compatible: cubic callers used a single center(i). Only valid when
    // the grid is cubic; asserts that via equal spacings.
    double center(std::size_t i) const { return (double(i)+0.5)*dx_; }

    std::size_t nx() const { return nx_; }
    std::size_t ny() const { return ny_; }
    std::size_t nz() const { return nz_; }
    std::size_t n()  const { return nx_; }            // legacy: cubic callers
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    double dz() const { return dz_; }

private:
    std::size_t idx(std::size_t i,std::size_t j,std::size_t k) const { return (i*ny_+j)*nz_+k; }
    std::vector<double> sampleCells(const ScalarField& f) const;
    void computeJoule(const std::vector<double>& sigma);

    std::size_t nx_, ny_, nz_;
    double Lx_, Ly_, Lz_, dx_, dy_, dz_;
    ScalarField sigmaField_;
    BCs ebc_;
    std::vector<double> V_, qv_;
};
}
