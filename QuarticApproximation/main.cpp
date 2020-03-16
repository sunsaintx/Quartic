#include "QuarticApproximation.h"
#include <iostream>
#include <complex>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
using namespace std;

/* int main()
{
    // QuarticApproximation(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_)
    int Nt_half = 16;
    int Ns = 4;
    double t = 1.;
    double T = 1.;
    double U = 3.;
    double delmutab[81];
    for (int i = 0; i < 81; i++)
        delmutab[i] = i * 0.25 - 10;
    ostringstream dir;
    dir << "../QAdata/Ns" << Ns << "/T" << T << "/U" << U << "/";
    ostringstream filename(dir.str(), ios::ate);
    filename << "Nt" << Nt_half * 2 << "_g(1,0)vsdelmu.dat";
    ofstream outfile(filename.str());
    if (!outfile)
    {
        cerr << "cannot open file \"" << filename.str() << "\"" << endl;
        exit(1);
    }
    QuarticApproximation mysystem(Nt_half, Ns, T, t, U, delmutab[0]);

    for (int delmuid = 0; delmuid < 81; delmuid++)
    {
        mysystem.setdelmu(delmutab[delmuid]);
        complex<double> gf = mysystem.greensfunction(Ns / 4, 0);
        outfile << setiosflags(ios::fixed) << setprecision(2) << setw(6) << delmutab[delmuid] << "    "
                << resetiosflags(ios::floatfield) << setiosflags(ios::scientific) << setprecision(6)
                << setw(13) << gf.real() << "    " << setw(13) << gf.imag() << resetiosflags(ios::floatfield) << endl;
    }
    return 0;
} */

//different U
int main()
{
    // QuarticApproximation(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_)
    int Nt_half = 16;
    int Ns = 4;
    double t = 1.;
    double T = 1.;
    double Utab[31];
    for (int i = 0; i < 31; i++)
        Utab[i] = i * T;
    //double delmu = 0;
    const double pi = 3.1415926535897932385;
    QuarticApproximation mysystem(Nt_half, Ns, T, t, 0, 0);

    for (int Uid = 0; Uid < 31; Uid++)
    {
        double delmu = 0.5 * Utab[Uid];
        ostringstream dir;
        dir << "../QAdata/Ns" << Ns << "/T" << T << "/U" << Utab[Uid] << "/delmu" << delmu << "/"
            << "Nt" << Nt_half * 2 << "/";
        cout << dir.str() << endl;
        system(("mkdir -p " + dir.str()).c_str());
        mysystem.setparms(Utab[Uid], delmu);
        for (int k = 0; k < Ns; k++)
        {
            ostringstream filename(dir.str(), ios::ate);
            filename << "g_k" << k << ".dat";
            cout << filename.str() << endl;
            ofstream greenfile(filename.str());
            if (!greenfile)
            {
                cerr << "cannot open file \"" << filename.str() << "\"" << endl;
                exit(1);
            }
            greenfile.setf(ios::scientific | ios::right);
            for (int n = 0; n < Nt_half; n++)
            {
                complex<double> gfkn = mysystem.greensfunction(k, n);
                greenfile << setw(13) << pi * T * (2. * n + 1) << "    "
                          << setw(13) << gfkn.real() << "    "
                          << setw(13) << gfkn.imag() << endl;
            }
            greenfile.close();
        }
        ostringstream filename(dir.str(), ios::ate);
        filename << "chi1.dat";
        cout << filename.str() << endl;
        ofstream spincorrfile(filename.str());
        if (!spincorrfile)
        {
            cerr << "cannot open file \"" << filename.str() << "\"" << endl;
            exit(1);
        }
        spincorrfile.setf(ios::scientific | ios::right);
        for (int n = 0; n < Nt_half; n++)
        {
            complex<double> chi = mysystem.spincorr(1, n);
            spincorrfile << setw(13) << pi * T * (2. * n) << "    "
                         << setw(13) << chi.real() << "    "
                         << setw(13) << chi.imag() << endl;
        }
        spincorrfile.close();
    }
    return 0;
}