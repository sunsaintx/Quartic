#ifndef QA_FERRO_H
#define QA_FERRO_H

#include <complex>
#include <algorithm>

class QA_Ferro
{
public:
    using Complex = std::complex<double>;
    QA_Ferro(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_, double h_);
    QA_Ferro(const QA_Ferro &) = delete;
    QA_Ferro &operator=(const QA_Ferro &) = delete;
    ~QA_Ferro();
    void setT(double T_);
    void sett(double t_);
    void setU(double U_);
    void setdelmu(double delmu_);
    void setparms(double U_, double delmu_);
    Complex greensfunction_up(int k, int n) const;   //1. Green's function of spinups
    Complex greensfunction_down(int k, int n) const; //2. Green's function of spindowns
    double nup(int k) const;                         //3. number of spinups
    double ndown(int k) const;                       //4. number of spindowns
    Complex denscorr(int idx, int n) const;          //5. density-density correlation(with frequency)
    Complex spincorr(int idx, int n) const;          //6. spin-spin correlation(with frequency)
private:
    void calculate();
    int vsum(int alpha, int beta, int gamma);
    static const Complex I; //imaginary unit
    static const double pi; //pi = 3.1415926535897932385;
    int Nt, Ns, Nts, NN;
    double T, t, U, delmu, mf, f_5;
    Complex *iw, *h, *g, *c1, *c2, *d1, *d2, *d3, *r1, *r2;
    int *sid;
    int *tid;
};

#endif