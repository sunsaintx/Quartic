#include "QuarticApproximation2D.h"
#include <iostream>
#include <complex>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
using namespace std;

int main()
{
    // QuarticApproximation(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_)
    int Nt_half = 128;
    int Nx = 1;
    int Ny = 4;
    double t = 1.;
    double T = 1.;
    double Utab[31];
    for (int i = 0; i < 31; i++)
        Utab[i] = i * T;
    //double delmu = 0;
    const double pi = 3.1415926535897932385;
    QuarticApproximation2D mysystem(Nt_half, Nx, Ny, T, t, 0, 0);

    for (int Uid = 0; Uid < 31; Uid++)
    {
        double delmu = 0 * Utab[Uid];
        ostringstream dir;
        dir << "../nQAdata/Nx" << Nx << "Ny" << Ny << "/T" << T << "/U" << Utab[Uid] << "/delmu" << delmu << "/"
            << "Nt" << Nt_half * 2 << "/";
        cout << dir.str() << endl;
        system(("mkdir -p " + dir.str()).c_str());
        mysystem.setparms(Utab[Uid], delmu);
        for (int kx = 0; kx < Nx; kx++)
            for (int ky = 0; ky < Ny; ky++)
            {
                ostringstream filename(dir.str(), ios::ate);
                filename << "g_kx" << kx << "ky" << ky << ".dat";
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
                    complex<double> gfkn = mysystem.greensfunction(kx, ky, n);
                    greenfile << setw(13) << pi * T * (2. * n + 1) << "    "
                              << setw(13) << gfkn.real() << "    "
                              << setw(13) << gfkn.imag() << endl;
                }
                greenfile.close();
            }
    }
    return 0;
}