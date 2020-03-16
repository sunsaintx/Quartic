#include "ED.h"
#include <iostream>
#include <complex>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
using namespace std;

//g(pi/2,0) vs delmu
/* int main()
{
    // ED::ED(int Ns_, double t_, double U_, double T_, double delmu_ = 0)
    int Ns = 4;
    int Nt = 64;
    double t = 1.;
    double T = 1.;
    double U = 10;
    double delmutab[101];
    for (int i = 0; i < 101; i++)
        delmutab[i] = i * 0.2 - 10;

    ostringstream dir;
    dir << "../EDdata/Ns" << Ns << "/T" << T << "/U" << U << "/";
    ofstream outfile(dir.str() + "g(1,0)vsdelmu.dat");
    if (!outfile)
    {
        cerr << "cannot open file \"" << dir.str() + "g(1,0)vsdelmu.dat"
             << "\"" << endl;
        exit(1);
    }
    outfile.setf(ios::right);
    for (int delmuid = 0; delmuid < 101; delmuid++)
    {
        ED mysystem(Ns, t, U, T, delmutab[delmuid]);
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
    // ED::ED(int Ns_, double t_, double U_, double T_, double delmu_ = 0)
    int Ns = 4;
    double t = 1.;
    double T = 1;
    double Utab[31];
    for (int i = 0; i < 31; i++)
        Utab[i] = i * T;
    //double delmu = 0.;
    const double pi = 3.1415926535897932385;
    const double eta = 0.005;

    int Nt = 128;
    vector<int> ntab;
    for (int n = 0; n < Nt; n++)
        ntab.push_back(n);
    for (int Uid = 0; Uid < 31; Uid++)
    {
        double delmu = 0.9 * Utab[Uid];
        ostringstream dir;
        dir << "../EDdata/Ns" << Ns << "/T" << T << "/U" << Utab[Uid] << "/delmu" << delmu << "/";
        cout << dir.str() << endl;
        system(("mkdir -p " + dir.str()).c_str());
        ED mysystem(Ns, t, Utab[Uid], T, delmu);
        for (int k = 0; k < Ns; k++)
        {
            ostringstream greenfilename(dir.str(), ios::ate);
            ostringstream specfilename(dir.str(), ios::ate);
            greenfilename << "g_k" << k << ".dat";
            specfilename << "spec_k" << k << ".dat";
            cout << greenfilename.str() << endl;
            cout << specfilename.str() << endl;
            ofstream greenfile(greenfilename.str());
            ofstream specfile(specfilename.str());
            if (!greenfile)
            {
                cerr << "cannot open file \"" << greenfilename.str() << "\"" << endl;
                exit(1);
            }
            if (!specfile)
            {
                cerr << "cannot open file \"" << specfilename.str() << "\"" << endl;
                exit(1);
            }

            greenfile.setf(ios::scientific | ios::right);
            vector<ED::Complex> gf = mysystem.greensfunction(k, ntab);
            for (int n = 0; n < Nt; n++)
            {
                greenfile << setw(13) << pi * T * (2. * ntab[n] + 1) << "    "
                          << setw(13) << gf[n].real() << "    "
                          << setw(13) << gf[n].imag() << endl;
            }
            greenfile.close();

            specfile.setf(ios::right);
            double range = std::max(Utab[Uid], 1.);
            double omega = -range;
            double step = eta / 5;
            vector<double> omegatab;
            while (omega <= range)
            {
                omegatab.push_back(omega);
                omega += step;
            }
            vector<double> spec = mysystem.spectralfunction(k, omegatab);
            for (int i = 0; i < spec.size(); i++)
            {
                specfile << setiosflags(ios::fixed) << setprecision(3) << setw(7) << omegatab[i] << "    "
                         << resetiosflags(ios::floatfield) << setiosflags(ios::scientific) << setprecision(6)
                         << spec[i] << resetiosflags(ios::floatfield) << endl;
            }
        }

        ostringstream chifilename(dir.str(), ios::ate);
        chifilename << "chi1.dat";
        cout << chifilename.str() << endl;
        ofstream spincorrfile(chifilename.str());
        if (!spincorrfile)
        {
            cerr << "cannot open file \"" << chifilename.str() << "\"" << endl;
            exit(1);
        }
        spincorrfile.setf(ios::scientific | ios::right);
        vector<ED::Complex> chi = mysystem.spincorr(1, ntab);
        for (int n = 0; n < Nt; n++)
        {
            spincorrfile << setw(13) << pi * T * (2. * n) << "    "
                         << setw(13) << chi[n].real() << "    "
                         << setw(13) << chi[n].imag() << endl;
        }
        spincorrfile.close();
    }
    return 0;
}