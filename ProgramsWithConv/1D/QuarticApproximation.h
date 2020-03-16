#ifndef QUARTIC_APPROXIMATION_H
#define QUARTIC_APPROXIMATION_H

#include <complex>
#include <algorithm>

class QuarticApproximation
{
public:
    using Complex = std::complex<double>;
    QuarticApproximation(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_);
    QuarticApproximation(const QuarticApproximation &) = delete;
    QuarticApproximation &operator=(const QuarticApproximation &) = delete;
    ~QuarticApproximation();
    void setT(double T_);
    void sett(double t_);
    void setU(double U_);
    void setdelmu(double delmu_);
    void setparms(double U_, double delmu_);
    Complex greensfunction(int k, int n) const; //1. Green's function
    //double nup(int k) const;                    //2. number of spinups
    //Complex denscorr(int idx, int n) const;     //3. density-density correlation(with frequency)
    Complex spincorr(int idx, int n) const; //4. spin-spin correlation(with frequency)

private:
    void calculate();
    int vsum(int alpha, int beta, int gamma);
    static const Complex I; //imaginary unit
    static const double pi; //pi = 3.1415926535897932385;
    int Nt, Ns, Nts, NN;
    double T, t, U, delmu, f_5;
    Complex *iw, *h, *g, *c, *d1, *d2, *r;
    int *sid;
    int *tid;
};

#endif