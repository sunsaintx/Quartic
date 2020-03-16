#ifndef QUARTIC_APPROXIMATION2D_H
#define QUARTIC_APPROXIMATION2D_H

#include <complex>

class QuarticApproximation2D
{
public:
    using Complex = std::complex<double>;
    QuarticApproximation2D(int Nt_half, int Nx_, int Ny_, double T_, double t_, double U_, double delmu_);
    QuarticApproximation2D(const QuarticApproximation2D &) = delete;
    QuarticApproximation2D &operator=(const QuarticApproximation2D &) = delete;
    ~QuarticApproximation2D();
    void setT(double T_);
    void sett(double t_);
    void setU(double U_);
    void setdelmu(double delmu_);
    void setparms(double U_, double delmu_);
    Complex greensfunction(int kx, int ky, int n) const;

private:
    void calculate();
    static const Complex I; //imaginary unit
    static const double pi; //pi = 3.1415926535897932385;
    int Nt, Nx, Ny, Ns, Nts, NN;
    double T, t, U, delmu, f_5;
    Complex *iw, *h, *g, *c, *d1, *d2, *r;
    int *xid, *yid;
    int *tid;
};

#endif