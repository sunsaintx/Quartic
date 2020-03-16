#include "QA_Ferro.h"

const QA_Ferro::Complex QA_Ferro::I(0, 1);
const double QA_Ferro::pi = 3.1415926535897932385;

QA_Ferro::QA_Ferro(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_, double h_)
    : Nt(Nt_half * 2), Ns(Ns_), Nts(Nt * Ns), NN(Nts * Nts), T(T_), t(t_), U(U_), delmu(delmu_), mf(h_), f_5(0.5 * U * T / Ns)
{
    iw = new Complex[Nt * 3];
    h = new Complex[Nts * 6];
    g = new Complex[Nts * 6];
    c1 = new Complex[NN * 2];
    c2 = new Complex[NN * 2];
    d1 = new Complex[NN * 2];
    d2 = new Complex[NN * 2];
    d3 = new Complex[NN * 2];
    r1 = new Complex[NN * 2];
    r2 = new Complex[NN * 2];

    sid = new int[3 * Nts];
    tid = new int[3 * Nts];

    for (int n = 0; n < 2 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - Nt_half) + 1);
    for (int n = 2 * Nt; n < 3 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - 3 * Nt - Nt_half) + 1);

    for (int alpha = 0; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu - mf);
        g[3 * Nts + alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu + mf);
    }
    //initialize chains
    for (int n = 0; n < 2 * NN; n++)
        c1[n] = 0., c2[n] = 0., d1[n] = 0., d2[n] = 0., d3[n] = 0., r1[n] = 0., r2[n] = 0.;

    for (int alpha = 0; alpha < 3 * Nts; alpha++)
    {
        tid[alpha] = alpha / Ns;
        sid[alpha] = alpha % Ns;
    }

    calculate();
}

void QA_Ferro::calculate()
{
    double p;
    if (U / T < 8)
        p = 0.3;
    else if (U / T >= 8 && U / T < 15)
        p = 0.2;
    else
        p = 0.1;
    Complex *dd = new Complex[2 * Nts];
    Complex *nc1 = new Complex[2 * NN];
    Complex *nc2 = new Complex[2 * NN];
    Complex *nd1 = new Complex[2 * NN];
    Complex *nd2 = new Complex[2 * NN];
    Complex *nd3 = new Complex[2 * NN];
    Complex *nr1 = new Complex[2 * NN];
    Complex *nr2 = new Complex[2 * NN];

    int Nit = 2000;
    for (int it = 0; it < Nit; it++)
    {
        Complex gsum[2] = {Complex(0., 0.), Complex(0., 0.)};
        for (int alpha = 0; alpha < 3 * Nts; alpha++)
            gsum[0] += g[alpha], gsum[1] += g[3 * Nts + alpha];
        gsum[0] *= T / Ns, gsum[1] *= T / Ns;

        for (int alpha = 0; alpha < 3 * Nts; alpha++)
        {
            h[alpha] = -1. / (iw[tid[alpha]] - 2. * t * cos(2. * pi * sid[alpha] / Ns) - delmu /*- U / 2.*/ - mf + U * gsum[1]);
            h[3 * Nts + alpha] = -1. / (iw[tid[alpha]] - 2. * t * cos(2. * pi * sid[alpha] / Ns) - delmu /*- U / 2.*/ + mf + U * gsum[0]);
        }

        for (int z = 0; z < 2; z++)
        {
            int b = 1 - z;
            int ginit_z = z * 3 * Nts, ginit_b = b * 3 * Nts;
            int chain_init_z = z * NN, chain_init_b = b * NN;
#pragma omp parallel for
            for (int beta = 0; beta < Nts; beta++)
            {
                for (int gamma = 0; gamma < Nts; gamma++)
                {
                    int iidx = Nts * beta + gamma;
                    //initialize new chains
                    nc1[chain_init_z + iidx] = nc2[chain_init_z + iidx] = nd1[chain_init_z + iidx] = nd2[chain_init_z + iidx] = nd3[chain_init_z + iidx] = nr1[chain_init_z + iidx] = nr2[chain_init_z + iidx] = 0.;
                    Complex divc1 = 1., divc2 = 1., divd1 = 1., divd2 = 1., divd3 = 1., divr1 = 1., divr2 = 1.;
                    for (int nu = 0; nu < Nts; nu++)
                    {
                        int jidx1 = beta * Nts + nu;
                        int jidx2 = gamma * Nts + nu;
                        int jidx3 = nu * Nts + beta;
                        int jidx4 = nu * Nts + gamma;

                        nd1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(gamma, nu, beta)] * g[ginit_z + gamma] * d2[chain_init_z + jidx1];
                        nd1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, nu, gamma)] * g[ginit_z + beta] * d2[chain_init_b + jidx4];
                        nd1[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * (g[ginit_b + vsum(beta, nu, gamma)] + g[ginit_b + vsum(gamma, nu, beta)]) * d3[chain_init_b + iidx];
                        nd1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(gamma, nu, beta)] * g[ginit_z + beta] * c1[chain_init_b + jidx4];
                        nd1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, nu, gamma)] * g[ginit_z + gamma] * r1[chain_init_b + jidx3];
                        nd1[chain_init_z + iidx] -= 2. * U * f_5 * h[ginit_b + nu] * g[ginit_z + beta] * g[ginit_z + gamma] * (g[ginit_b + vsum(beta, nu, gamma)] + g[ginit_b + vsum(gamma, nu, beta)]);

                        //nd2[chain_init_z + iidx] -= f_5 * (h[ginit_z + nu] * g[ginit_b + vsum(gamma, nu, beta)] + h[ginit_b + nu] * g[ginit_z + vsum(beta, nu, gamma)]) * d2[chain_init_z + iidx];
                        nd2[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(gamma, nu, beta)] * g[ginit_b + gamma] * d3[chain_init_b + jidx1];
                        nd2[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, nu, gamma)] * g[ginit_z + beta] * d3[chain_init_z + jidx4];
                        nd2[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(gamma, nu, beta)] * g[ginit_z + beta] * c2[chain_init_b + jidx4];
                        nd2[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, nu, gamma)] * g[ginit_b + gamma] * r2[chain_init_z + jidx3];
                        nd2[chain_init_z + iidx] -= 2. * U * f_5 * g[ginit_z + beta] * g[ginit_b + gamma] * (h[ginit_b + nu] * g[ginit_z + vsum(beta, nu, gamma)] + h[ginit_z + nu] * g[ginit_b + vsum(gamma, nu, beta)]);

                        nd3[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(gamma, nu, beta)] * g[ginit_b + gamma] * d1[chain_init_b + jidx1];
                        nd3[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, nu, gamma)] * g[ginit_b + beta] * d1[chain_init_b + jidx4];
                        nd3[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * (g[ginit_b + vsum(gamma, nu, beta)] + g[ginit_b + vsum(beta, nu, gamma)]) * d1[chain_init_b + iidx];

                        nc1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_z + beta] * d2[chain_init_z + jidx4];
                        nc1[chain_init_z + iidx] -= f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_b + gamma] * d2[chain_init_b + jidx3];
                        nc1[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_z + beta] * d3[chain_init_z + jidx4];
                        nc1[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_b + gamma] * d3[chain_init_b + jidx3];
                        //nc1[chain_init_z + iidx] -= f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)] * c1[chain_init_z + iidx];
                        nc1[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] * c2[chain_init_b + iidx];
                        nc1[chain_init_z + iidx] -= 2. * U * f_5 * g[ginit_z + beta] * g[ginit_b + gamma] * (h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] + h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)]);

                        nc2[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_z + gamma] * d2[chain_init_z + jidx3];
                        nc2[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_b + beta] * d2[chain_init_b + jidx4];
                        nc2[chain_init_z + iidx] -= f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_z + gamma] * d3[chain_init_z + jidx3];
                        nc2[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_b + beta] * d3[chain_init_b + jidx4];
                        //nc2[chain_init_z + iidx] -= f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)] * c2[chain_init_z + iidx];
                        nc2[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] * c1[chain_init_b + iidx];
                        nc2[chain_init_z + iidx] += 2. * U * f_5 * g[ginit_b + beta] * g[ginit_z + gamma] * (h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] + h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)]);

                        nr1[chain_init_z + iidx] -= f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_b + gamma] * d2[chain_init_z + jidx1];
                        nr1[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_z + beta] * d2[chain_init_b + jidx2];
                        nr1[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_z + beta] * d3[chain_init_z + jidx2];
                        nr1[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_b + gamma] * d3[chain_init_b + jidx1];
                        //nr1[chain_init_z + iidx] -= f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)] * r1[chain_init_z + iidx];
                        nr1[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] * r2[chain_init_b + iidx];
                        nr1[chain_init_z + iidx] -= 2. * U * f_5 * g[ginit_z + beta] * g[ginit_b + gamma] * (h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] + h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)]);

                        nr2[chain_init_z + iidx] += f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_b + beta] * d2[chain_init_z + jidx2];
                        nr2[chain_init_z + iidx] += f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_z + gamma] * d2[chain_init_b + jidx1];
                        nr2[chain_init_z + iidx] -= f_5 * h[ginit_z + vsum(beta, gamma, nu)] * g[ginit_z + gamma] * d3[chain_init_z + jidx1];
                        nr2[chain_init_z + iidx] -= f_5 * h[ginit_b + vsum(beta, gamma, nu)] * g[ginit_b + beta] * d3[chain_init_b + jidx2];
                        nr2[chain_init_z + iidx] += f_5 * h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] * r1[chain_init_b + iidx];
                        //nr2[chain_init_z + iidx] -= f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)] * r2[chain_init_z + iidx];
                        nr2[chain_init_z + iidx] += 2. * U * f_5 * g[ginit_b + beta] * g[ginit_z + gamma] * (h[ginit_b + nu] * g[ginit_z + vsum(beta, gamma, nu)] + h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)]);

                        divd2 += f_5 * (h[ginit_z + nu] * g[ginit_b + vsum(gamma, nu, beta)] + h[ginit_b + nu] * g[ginit_z + vsum(beta, nu, gamma)]);
                        divc1 += f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)];
                        divc2 += f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)];
                        divr1 += f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)];
                        divr2 += f_5 * h[ginit_z + nu] * g[ginit_b + vsum(beta, gamma, nu)];
                    }
                    nd2[chain_init_z + iidx] /= divd2;
                    nc1[chain_init_z + iidx] /= divc1;
                    nc2[chain_init_z + iidx] /= divc2;
                    nr1[chain_init_z + iidx] /= divr1;
                    nr2[chain_init_z + iidx] /= divr2;
                }
            }
        }

        double chain_diff = 0.;
        for (int i = 0; i < 2 * NN; i++)
        {
            chain_diff = std::max(chain_diff, std::abs(nc1[i] - c1[i]));
            chain_diff = std::max(chain_diff, std::abs(nc2[i] - c2[i]));
            chain_diff = std::max(chain_diff, std::abs(nd1[i] - d1[i]));
            chain_diff = std::max(chain_diff, std::abs(nd2[i] - d2[i]));
            chain_diff = std::max(chain_diff, std::abs(nd3[i] - d3[i]));
            chain_diff = std::max(chain_diff, std::abs(nr1[i] - r1[i]));
            chain_diff = std::max(chain_diff, std::abs(nr2[i] - r2[i]));
        }

        for (int n = 0; n < 2 * NN; n++)
            c1[n] = nc1[n], c2[n] = nc2[n], d1[n] = nd1[n], d2[n] = nd2[n], d3[n] = nd3[n], r1[n] = nr1[n], r2[n] = nr2[n];

        for (int alpha = 0; alpha < Nts; alpha++)
        {
            dd[alpha] = 0, dd[Nts + alpha] = 0.;
            for (int nu = 0; nu < Nts; nu++)
            {
                dd[alpha] += r1[Nts * alpha + nu] + d1[Nts * alpha + nu] + d2[Nts * alpha + nu];
                dd[Nts + alpha] += r1[NN + Nts * alpha + nu] + d1[NN + Nts * alpha + nu] + d2[NN + Nts * alpha + nu];
            }
            dd[alpha] = (T / (6. * Ns)) * dd[alpha];
            dd[Nts + alpha] = (T / (6. * Ns)) * dd[Nts + alpha];
        }

        //get new g[n]
        double green_diff = 0.;
        for (int alpha = 0; alpha < Nts; alpha++)
        {
            //g[alpha] = h[alpha]*(1.+dd[alpha]);
            Complex newg_alpha_up = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
            Complex newg_alpha_down = h[3 * Nts + alpha] / (1. - h[3 * Nts + alpha] * dd[Nts + alpha] / g[3 * Nts + alpha]);
            //dd[alpha] = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
            newg_alpha_up = p * newg_alpha_up + (1. - p) * g[alpha];
            newg_alpha_down = p * newg_alpha_down + (1. - p) * g[3 * Nts + alpha];
            green_diff = std::max(green_diff, std::abs(newg_alpha_up - g[alpha]));
            green_diff = std::max(green_diff, std::abs(newg_alpha_down - g[3 * Nts + alpha]));
            g[alpha] = newg_alpha_up;
            g[3 * Nts + alpha] = newg_alpha_down;
        }

        //std::cout << "gup(0,0) = " << g[Nt / 2 * Ns].real() << "+I*" << g[Nt / 2 * Ns].imag() << std::endl;
        //std::cout << "gdown(0,0) = " << g[3 * Nts + Nt / 2 * Ns].real() << "+I*" << g[3 * Nts + Nt / 2 * Ns].imag() << std::endl;
        if (chain_diff < 1e-6 && green_diff < 1e-7)
            break;
    }

    delete[] dd;
    delete[] nc1;
    delete[] nc2;
    delete[] nd1;
    delete[] nd2;
    delete[] nd3;
    delete[] nr1;
    delete[] nr2;
}

void QA_Ferro::setT(double T_)
{
    T = T_;
    f_5 = 0.5 * U * T / Ns;
    for (int n = 0; n < 2 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - Nt / 2) + 1);
    for (int n = 2 * Nt; n < 3 * Nt; n++)
        iw[n] = I * pi * T * (2. * (n - 3 * Nt - Nt / 2) + 1);
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu - mf);
        g[3 * Nts + alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu + mf);
    }
    calculate();
}

void QA_Ferro::sett(double t_)
{
    t = t_;
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu - mf);
        g[3 * Nts + alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu + mf);
    }
    calculate();
}

void QA_Ferro::setU(double U_)
{
    U = U_;
    f_5 = 0.5 * U * T / Ns;
    calculate();
}

void QA_Ferro::setdelmu(double delmu_)
{
    delmu = delmu_;
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu - mf);
        g[3 * Nts + alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu + mf);
    }
    calculate();
}

void QA_Ferro::setparms(double U_, double delmu_)
{
    U = U_;
    delmu = delmu_;
    f_5 = 0.5 * U * T / Ns;
    for (int alpha = Nts; alpha < 3 * Nts; alpha++)
    {
        int omega = alpha / Ns, k = alpha % Ns;
        g[alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu - mf);
        g[3 * Nts + alpha] = -1. / (iw[omega] - 2. * t * cos(2. * pi * k / Ns) - delmu + mf);
    }
    calculate();
}

QA_Ferro::Complex QA_Ferro::greensfunction_up(int k, int n) const
{
    if (n > 3 * Nt / 2 - 1 || n < -3 * Nt / 2)
        return 0;
    k = k % Ns;
    int tid = n + Nt / 2 < 0 ? n + 7 * Nt / 2 : n + Nt / 2;
    return g[tid * Ns + k];
}

QA_Ferro::Complex QA_Ferro::greensfunction_down(int k, int n) const
{
    if (n > 3 * Nt / 2 - 1 || n < -3 * Nt / 2)
        return 0;
    k = k % Ns;
    int tid = n + Nt / 2 < 0 ? n + 7 * Nt / 2 : n + Nt / 2;
    return g[3 * Nts + tid * Ns + k];
}

inline int QA_Ferro::vsum(int alpha, int beta, int gamma) //++-
{
    return (tid[alpha] + tid[beta] - tid[gamma] + 3 * Nt) % (3 * Nt) * Ns + (sid[alpha] + sid[beta] - sid[gamma] + Ns) % Ns;
}

QA_Ferro::~QA_Ferro()
{
    delete[] iw;
    delete[] h;
    delete[] g;
    delete[] c1;
    delete[] c2;
    delete[] d1;
    delete[] d2;
    delete[] d3;
    delete[] r1;
    delete[] r2;
    delete[] tid;
    delete[] sid;
}
