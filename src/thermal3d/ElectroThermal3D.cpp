#include "thermal3d/ElectroThermal3D.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace physi_sim::thermal3d {

namespace {
double optimalOmega(std::size_t nmax) {
    const double pi=3.14159265358979323846;
    return 2.0/(1.0+std::sin(pi/double(nmax)));
}
}

ElectroThermal3D::ElectroThermal3D(std::size_t nx, std::size_t ny, std::size_t nz,
                                   double Lx, double Ly, double Lz, ScalarField sigma)
    : nx_(nx), ny_(ny), nz_(nz), Lx_(Lx), Ly_(Ly), Lz_(Lz),
      dx_(Lx/double(nx)), dy_(Ly/double(ny)), dz_(Lz/double(nz)),
      sigmaField_(std::move(sigma)),
      V_(nx*ny*nz,0.0), qv_(nx*ny*nz,0.0) {
    if (nx<2||ny<2||nz<2) throw std::invalid_argument("ElectroThermal3D: need >=2 cells per axis");
    if (Lx<=0||Ly<=0||Lz<=0) throw std::invalid_argument("ElectroThermal3D: lengths must be > 0");
    for (std::size_t f=0; f<FaceCount; ++f) {
        ebc_[f].type=FaceBC::Neumann; ebc_[f].value=[](double,double,double){return 0.0;}; ebc_[f].h=0.0;
    }
}

void ElectroThermal3D::setElectricalBC(const BCs& bc){ ebc_=bc; }

std::vector<double> ElectroThermal3D::sampleCells(const ScalarField& f) const {
    std::vector<double> d(nx_*ny_*nz_);
    for (std::size_t i=0;i<nx_;++i) for (std::size_t j=0;j<ny_;++j) for (std::size_t k=0;k<nz_;++k) {
        double v=f(centerX(i),centerY(j),centerZ(k));
        if (v<=0.0) throw std::invalid_argument("material field must be strictly positive");
        d[idx(i,j,k)]=v;
    }
    return d;
}

// ---------------------------------------------------------------------------
//  solveField -- div(gamma grad phi) + src = 0, ANISOTROPIC.
//  Each face now uses the AREA normal to its axis and the SPACING along its axis:
//     x-face: area dy*dz, spacing dx     y-face: area dx*dz, spacing dy
//     z-face: area dx*dy, spacing dz
//  Conductance G = gamma_face * area / spacing. In the cubic case all three
//  reduce to gamma*dx and this is identical to the old code -- which is why the
//  cubic ladder stays exact.
// ---------------------------------------------------------------------------
bool ElectroThermal3D::solveField(const std::vector<double>& g, const BCs& bc,
                                  const std::vector<double>& src, std::vector<double>& phi,
                                  double tol, std::size_t maxIter) {
    const double Ax=dy_*dz_, Ay=dx_*dz_, Az=dx_*dy_;   // face areas per axis
    const double vol=dx_*dy_*dz_;

    // interior face conductance: harmonic mean * area / spacing
    auto gface=[&](double gP,double gN,double A,double d){ return (2.0*gP*gN/(gP+gN))*A/d; };
    // boundary: returns {diag add, rhs add}
    auto bcAdd=[&](const FaceBC& f,double gP,double A,double d,double fx,double fy,double fz)
        -> std::pair<double,double> {
        if (f.type==FaceBC::Dirichlet){ double G=gP*A/(d/2.0); return {G,G*f.value(fx,fy,fz)}; }
        if (f.type==FaceBC::Robin){ double U=A/(d/(2.0*gP)+1.0/f.h); return {U,U*f.value(fx,fy,fz)}; }
        return {0.0, f.value(fx,fy,fz)*A};   // Neumann: inward flux * area
    };

    const std::size_t sx=ny_*nz_, sy=nz_, sz=1;
    const double omega=optimalOmega(std::max({nx_,ny_,nz_}));

    for (std::size_t it=0; it<maxIter; ++it) {
        double md=0.0;
        for (std::size_t i=0;i<nx_;++i) for (std::size_t j=0;j<ny_;++j) for (std::size_t k=0;k<nz_;++k) {
            const std::size_t c=idx(i,j,k);
            const double x=centerX(i),y=centerY(j),z=centerZ(k),gc=g[c];
            double diag=0.0, rhs=src[c]*vol;

            if(i>0){ double G=gface(gc,g[c-sx],Ax,dx_); diag+=G; rhs+=G*phi[c-sx]; }
            else   { auto[G,r]=bcAdd(bc[XLow], gc,Ax,dx_,0.0,y,z); diag+=G; rhs+=r; }
            if(i+1<nx_){ double G=gface(gc,g[c+sx],Ax,dx_); diag+=G; rhs+=G*phi[c+sx]; }
            else   { auto[G,r]=bcAdd(bc[XHigh],gc,Ax,dx_,Lx_,y,z); diag+=G; rhs+=r; }
            if(j>0){ double G=gface(gc,g[c-sy],Ay,dy_); diag+=G; rhs+=G*phi[c-sy]; }
            else   { auto[G,r]=bcAdd(bc[YLow], gc,Ay,dy_,x,0.0,z); diag+=G; rhs+=r; }
            if(j+1<ny_){ double G=gface(gc,g[c+sy],Ay,dy_); diag+=G; rhs+=G*phi[c+sy]; }
            else   { auto[G,r]=bcAdd(bc[YHigh],gc,Ay,dy_,x,Ly_,z); diag+=G; rhs+=r; }
            if(k>0){ double G=gface(gc,g[c-sz],Az,dz_); diag+=G; rhs+=G*phi[c-sz]; }
            else   { auto[G,r]=bcAdd(bc[ZLow], gc,Az,dz_,x,y,0.0); diag+=G; rhs+=r; }
            if(k+1<nz_){ double G=gface(gc,g[c+sz],Az,dz_); diag+=G; rhs+=G*phi[c+sz]; }
            else   { auto[G,r]=bcAdd(bc[ZHigh],gc,Az,dz_,x,y,Lz_); diag+=G; rhs+=r; }

            if(diag==0.0) throw std::runtime_error("singular: zero diagonal (all-Neumann cell)");
            double d=omega*(rhs/diag - phi[c]);
            phi[c]+=d; md=std::max(md,std::abs(d));
        }
        if(md<tol) return true;
    }
    return false;
}

void ElectroThermal3D::computeJoule(const std::vector<double>& sigma) {
    const double Ax=dy_*dz_, Ay=dx_*dz_, Az=dx_*dy_, vol=dx_*dy_*dz_;
    const std::size_t sx=ny_*nz_, sy=nz_, sz=1;
    auto gface=[&](double a,double b,double A,double d){ return (2.0*a*b/(a+b))*A/d; };
    for (std::size_t i=0;i<nx_;++i) for (std::size_t j=0;j<ny_;++j) for (std::size_t k=0;k<nz_;++k) {
        const std::size_t c=idx(i,j,k); const double sc=sigma[c],Vc=V_[c]; double P=0.0;
        auto branch=[&](std::size_t m,double A,double d){ double G=gface(sc,sigma[m],A,d),dV=Vc-V_[m];
            P+=G*dV*dV*(sigma[m]/(sc+sigma[m])); };
        auto dir=[&](Face fc,double A,double d,double fx,double fy,double fz){
            if(ebc_[fc].type==FaceBC::Dirichlet){ double G=sc*A/(d/2.0),dV=Vc-ebc_[fc].value(fx,fy,fz); P+=G*dV*dV; } };
        const double x=centerX(i),y=centerY(j),z=centerZ(k);
        if(i>0) branch(c-sx,Ax,dx_); else dir(XLow,Ax,dx_,0.0,y,z);
        if(i+1<nx_) branch(c+sx,Ax,dx_); else dir(XHigh,Ax,dx_,Lx_,y,z);
        if(j>0) branch(c-sy,Ay,dy_); else dir(YLow,Ay,dy_,x,0.0,z);
        if(j+1<ny_) branch(c+sy,Ay,dy_); else dir(YHigh,Ay,dy_,x,Ly_,z);
        if(k>0) branch(c-sz,Az,dz_); else dir(ZLow,Az,dz_,x,y,0.0);
        if(k+1<nz_) branch(c+sz,Az,dz_); else dir(ZHigh,Az,dz_,x,y,Lz_);
        qv_[c]=P/vol;
    }
}

bool ElectroThermal3D::solve(double tol, std::size_t maxIter) {
    auto sigma=sampleCells(sigmaField_);
    std::vector<double> zero(nx_*ny_*nz_,0.0);
    bool ok=solveField(sigma,ebc_,zero,V_,tol,maxIter);
    if(ok) computeJoule(sigma);
    return ok;
}

bool ElectroThermal3D::solveThermal(const ScalarField& k_field, const BCs& bcs,
                                    const std::vector<double>& src, std::vector<double>& T, double tol) {
    auto g=sampleCells(k_field);
    return solveField(g,bcs,src,T,tol,500000);
}

double ElectroThermal3D::voltage(std::size_t i,std::size_t j,std::size_t k) const { return V_[idx(i,j,k)]; }
double ElectroThermal3D::jouleDensity(std::size_t i,std::size_t j,std::size_t k) const { return qv_[idx(i,j,k)]; }
double ElectroThermal3D::totalJoulePower() const { double s=0,v=dx_*dy_*dz_; for(double q:qv_)s+=q*v; return s; }
}
