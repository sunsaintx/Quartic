#include "QuarticApproximation.h"

const QuarticApproximation::Complex QuarticApproximation::I(0, 1);
const double QuarticApproximation::pi = 3.1415926535897932385;

QuarticApproximation::QuarticApproximation(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_)
    : Nt(Nt_half * 2), Ns(Ns_), Nts(Nt * Ns_), NN(Nts * Nts), T(T_), t(t_), U(U_), delmu(delmu_), f_5(0.5 * U * T / Ns)
{
    iw = new Complex[Nt * 3];
    h = new Complex[Nts * 3];
    g = new Complex[Nts * 3];
    c = new Complex[NN];
    d1 = new Complex[NN];
    d2 = new Complex[NN];
    r = new Complex[NN];

    sid = new int[3 * Nts];
    tid = new int[3 * Nts];

    for (int n = 0; n < 2 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - Nt_half) + 1);
    for (int n = 2 * Nt; n < 3 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - 3 * Nt - Nt_half) + 1);

    for (int alpha = 0; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu);
    }

    //initialize chains
    for (int n = 0; n < NN; n++)
        c[n] = 0., d1[n] = 0., d2[n] = 0., r[n] = 0.;

    for (int alpha = 0; alpha < 3 * Nts; alpha++)
    {
        tid[alpha] = alpha / Ns;
        sid[alpha] = alpha % Ns;
    }

    calculate();
}

void QuarticApproximation::calculate()
{
    double p;
    if (U / T < 8)
        p = 0.3;
    else if (U / T >= 8 && U / T < 15)
        p = 0.1;
    else
        p = 0.05;
    Complex *dd = new Complex[Nts];
    Complex *nc = new Complex[NN];
    Complex *nd1 = new Complex[NN];
    Complex *nd2 = new Complex[NN];
    Complex *nr = new Complex[NN];
    int Nit = 2000;
    for (int it = 0; it < Nit; it++)
    {
        Complex nhalf = 0;
        for (int alpha = 0; alpha < 3 * Nts; alpha++)
            nhalf += g[alpha];
        nhalf *= T / Ns;

        for (int alpha = 0; alpha < 3 * Nts; alpha++)
        {
            int omega = alpha / Ns, k = alpha % Ns;
            h[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu /*- U / 2.*/ + U * nhalf);
        }

#pragma omp parallel for
        for (int beta = 0; beta < Nts; beta++)
        {
            for (int gamma = 0; gamma < Nts; gamma++)
            {
                int iidx = Nts * beta + gamma;
                //initialize new chains
                nc[Nts * beta + gamma] = 0., nd1[Nts * beta + gamma] = 0., nd2[Nts * beta + gamma] = 0., nr[Nts * beta + gamma] = 0.;
                Complex divc = 1., divd1 = 1., divd2 = 1., divr = 1.; //divisors
                for (int nu = 0; nu < Nts; nu++)
                {
                    int jidx1 = beta * Nts + nu;
                    int jidx2 = gamma * Nts + nu;
                    int jidx3 = nu * Nts + beta;
                    int jidx4 = nu * Nts + gamma;

                    Complex prod1 = f_5 * h[vsum(beta, gamma, nu)] * g[beta];
                    Complex prod2 = f_5 * h[vsum(beta, gamma, nu)] * g[gamma];
                    Complex prod3 = f_5 * h[vsum(beta, nu, gamma)] * g[beta];
                    Complex prod4 = f_5 * h[vsum(gamma, nu, beta)] * g[gamma];
                    Complex prod5 = f_5 * h[vsum(gamma, nu, beta)] * g[beta];
                    Complex prod6 = f_5 * h[vsum(beta, nu, gamma)] * g[gamma];

                    Complex prod7 = f_5 * h[nu] * g[vsum(beta, gamma, nu)];
                    Complex prod8 = f_5 * h[nu] * (g[vsum(beta, nu, gamma)] + g[vsum(gamma, nu, beta)]);

                    //nc[Nts * beta + gamma] -= f_5 * h[vsum(beta, gamma, nu)] * g[beta] * (d2[Nts * nu + gamma] - d1[Nts * nu + gamma]);
                    nc[iidx] -= prod1 * (d2[jidx4] - d1[jidx4]);
                    //nc[Nts * beta + gamma] -= f_5 * h[vsum(beta, gamma, nu)] * g[gamma] * (d2[Nts * nu + beta] - d1[Nts * nu + beta]);
                    nc[iidx] -= prod2 * (d2[jidx3] - d1[jidx3]);
                    //nc[Nts * beta + gamma] -= 4. * U * f_5 * h[nu] * g[beta] * g[gamma] * g[vsum(beta, gamma, nu)];
                    nc[iidx] -= 4. * U * prod7 * g[beta] * g[gamma];
                    //nd1[Nts * beta + gamma] += f_5 * h[nu] * (g[vsum(gamma, nu, beta)] + g[vsum(beta, nu, gamma)]) * d2[Nts * beta + gamma];
                    nd1[iidx] += prod8 * d2[iidx];
                    //nd1[Nts * beta + gamma] -= f_5 * h[vsum(beta, nu, gamma)] * g[beta] * (d1[Nts * nu + gamma] + d2[Nts * nu + gamma]);
                    nd1[iidx] -= prod3 * (d1[jidx4] + d2[jidx4]);
                    //nd1[Nts * beta + gamma] -= f_5 * h[vsum(gamma, nu, beta)] * g[gamma] * (d1[Nts * beta + nu] + d2[Nts * beta + nu]);
                    nd1[iidx] -= prod4 * (d1[jidx1] + d2[jidx1]);
                    //nd2[Nts * beta + gamma] += f_5 * h[vsum(beta, nu, gamma)] * g[beta] * d1[Nts * nu + gamma];
                    nd2[iidx] += prod3 * d1[jidx4];
                    //nd2[Nts * beta + gamma] += f_5 * h[vsum(gamma, nu, beta)] * g[gamma] * d1[Nts * beta + nu];
                    nd2[iidx] += prod4 * d1[jidx1];
                    //nd2[Nts * beta + gamma] -= f_5 * h[vsum(gamma, nu, beta)] * g[beta] * c[Nts * nu + gamma];
                    nd2[iidx] -= prod5 * c[jidx4];
                    //nd2[Nts * beta + gamma] -= f_5 * h[vsum(beta, nu, gamma)] * g[gamma] * r[Nts * nu + beta];
                    nd2[iidx] -= prod6 * r[jidx3];
                    //nd2[Nts * beta + gamma] -= 2. * U * f_5 * h[nu] * g[beta] * g[gamma] * (g[vsum(beta, nu, gamma)] + g[vsum(gamma, nu, beta)]);
                    nd2[iidx] -= 2. * U * prod8 * g[beta] * g[gamma];
                    //nr[Nts * beta + gamma] -= f_5 * h[vsum(beta, gamma, nu)] * g[beta] * (d2[Nts * gamma + nu] - d1[Nts * gamma + nu]);
                    nr[iidx] -= prod1 * (d2[jidx2] - d1[jidx2]);
                    //nr[Nts * beta + gamma] -= f_5 * h[vsum(beta, gamma, nu)] * g[gamma] * (d2[Nts * beta + nu] - d1[Nts * beta + nu]);
                    nr[iidx] -= prod2 * (d2[jidx1] - d1[jidx1]);
                    //nr[Nts * beta + gamma] -= 4. * U * f_5 * h[nu] * g[beta] * g[gamma] * g[vsum(beta, gamma, nu)];
                    nr[iidx] -= 4. * U * prod7 * g[beta] * g[gamma];

                    //divisors
                    //divc += 2. * f_5 * h[nu] * g[vsum(beta, gamma, nu)];
                    divc += 2. * prod7;
                    //divd1 -= f_5 * h[nu] * (g[vsum(gamma, nu, beta)] + g[vsum(beta, nu, gamma)]);
                    divd1 -= prod8;
                    //divd2 += f_5 * h[nu] * (g[vsum(gamma, nu, beta)] + g[vsum(beta, nu, gamma)]);
                    divd2 += prod8;
                    //divr += 2. * f_5 * h[nu] * g[vsum(beta, gamma, nu)];
                    divr += 2. * prod7;
                }

                //nc[Nts * beta + gamma] /= divc;
                //nd1[Nts * beta + gamma] /= divd1;
                //nd2[Nts * beta + gamma] /= divd2;
                //nr[Nts * beta + gamma] /= divr;

                nc[iidx] /= divc;
                nd1[iidx] /= divd1;
                nd2[iidx] /= divd2;
                nr[iidx] /= divr;
            }
        }

        double chain_diff = 0.;
        for (int i = 0; i < NN; i++)
        {
            chain_diff = std::max(chain_diff, std::abs(nc[i] - c[i]));
            chain_diff = std::max(chain_diff, std::abs(nd1[i] - d1[i]));
            chain_diff = std::max(chain_diff, std::abs(nd2[i] - d2[i]));
            chain_diff = std::max(chain_diff, std::abs(nr[i] - r[i]));
        }

        for (int n = 0; n < NN; n++)
            c[n] = nc[n], d1[n] = nd1[n], d2[n] = nd2[n], r[n] = nr[n];

        for (int alpha = 0; alpha < Nts; alpha++)
        {
            dd[alpha] = 0;
            for (int nu = 0; nu < Nts; nu++)
                dd[alpha] += r[Nts * alpha + nu] + d1[Nts * alpha + nu] + 2. * d2[Nts * alpha + nu];
            dd[alpha] = (T / (6. * Ns)) * dd[alpha];
        }

        //get new g[n]
        double green_diff = 0.;
        for (int alpha = 0; alpha < Nts; alpha++)
        {
            //g[alpha] = h[alpha]*(1.+dd[alpha]);
            Complex newg_alpha = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
            //dd[alpha] = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
            newg_alpha = p * newg_alpha + (1. - p) * g[alpha];
            green_diff = std::max(green_diff, std::abs(newg_alpha - g[alpha]));
            g[alpha] = newg_alpha;
        }

        //std::cout << "g(0,0) = " << g[Nt / 2 * Ns].real() << "+I*" << g[Nt / 2 * Ns].imag() << std::endl;
        if (chain_diff < 1e-6 && green_diff < 1e-7)
            break;
    }

    delete[] dd;
    delete[] nc;
    delete[] nd1;
    delete[] nd2;
    delete[] nr;
}

void QuarticApproximation::setT(double T_)
{
    T = T_;
    f_5 = 0.5 * U * T / Ns;
    calculate();
}

void QuarticApproximation::sett(double t_)
{
    t = t_;
    calculate();
}

void QuarticApproximation::setU(double U_)
{
    U = U_;
    f_5 = 0.5 * U * T / Ns;
    calculate();
}

void QuarticApproximation::setdelmu(double delmu_)
{
    delmu = delmu_;
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu);
    }
    calculate();
}

void QuarticApproximation::setparms(double U_, double delmu_)
{
    U = U_;
    delmu = delmu_;
    f_5 = 0.5 * U * T / Ns;
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu);
    }
    calculate();
}

QuarticApproximation::Complex QuarticApproximation::greensfunction(int k, int n) const
{
    if (n > 3 * Nt / 2 - 1 || n < -3 * Nt / 2)
        return 0;
    k = k % Ns;
    int tid = n + Nt / 2 < 0 ? n + 7 * Nt / 2 : n + Nt / 2;
    return g[tid * Ns + k];
}

/* QuarticApproximation::Complex QuarticApproximation::spincorr(int idx, int n) const
{
    Complex chi = 0;
    for (int gamma = 0; gamma < Nts; gamma++)
    {
        for (int beta_k = 0; beta_k < Ns; beta_k++)
        {
            int beta_omega = tid[gamma] - Nt / 2 + n;
            if (beta_omega > 3 * Nt / 2 - 1 || beta_omega < -3 * Nt / 2)
                continue;
            int beta_tid = beta_omega + Nt / 2 < 0 ? beta_omega + 7 * Nt / 2 : beta_omega + Nt / 2;
            int beta = beta_tid * Ns + beta_k;
            if (beta_tid >= 2 * Nt)
                chi += std::exp(2. * I * pi * double(idx) * double(beta_k - sid[gamma]) / double(Ns)) * 2. * g[beta] * g[gamma];
            else
                chi += std::exp(2. * I * pi * double(idx) * double(beta_k - sid[gamma]) / double(Ns)) * (d2[Nts * beta + gamma] / U + 2. * g[beta] * g[gamma]);
        }
    }
    return -chi * T / double(Ns * Ns);
} */

QuarticApproximation::Complex QuarticApproximation::spincorr(int idx, int n) const
{
    Complex chi = 0;
    double rchi = 0., ichi = 0.;
#pragma omp parallel for reduction(+ \
                                   : rchi, ichi)
    for (int gamma = 0; gamma < 3 * Nts; gamma++)
    {
        Complex temp(0, 0);
        for (int beta_k = 0; beta_k < Ns; beta_k++)
        {
            int beta_omega = (tid[gamma] >= 2 * Nt ? tid[gamma] - 7 * Nt / 2 : tid[gamma] - Nt / 2) - n;
            if (beta_omega > 3 * Nt / 2 - 1 || beta_omega < -3 * Nt / 2)
                continue;
            int beta_tid = beta_omega + Nt / 2 < 0 ? beta_omega + 7 * Nt / 2 : beta_omega + Nt / 2;
            int beta = beta_tid * Ns + beta_k;
            if (tid[gamma] >= Nt || beta_tid >= Nt || U == 0.)
                temp += std::exp(2. * I * pi * double(idx) * double(beta_k - sid[gamma]) / double(Ns)) * 2. * g[beta] * g[gamma];
            else
                temp += std::exp(2. * I * pi * double(idx) * double(beta_k - sid[gamma]) / double(Ns)) * (d2[Nts * beta + gamma] / U + 2. * g[beta] * g[gamma]);
        }
        rchi += temp.real();
        ichi += temp.imag();
    }
    chi = Complex(rchi, ichi);
    return -chi * T / double(Ns * Ns);
}

inline int QuarticApproximation::vsum(int alpha, int beta, int gamma) //++-
{
    return (tid[alpha] + tid[beta] - tid[gamma] + 3 * Nt) % (3 * Nt) * Ns + (sid[alpha] + sid[beta] - sid[gamma] + Ns) % Ns;
}

QuarticApproximation::~QuarticApproximation()
{
    delete[] iw;
    delete[] h;
    delete[] g;
    delete[] c;
    delete[] d1;
    delete[] d2;
    delete[] r;
    delete[] tid;
    delete[] sid;
}