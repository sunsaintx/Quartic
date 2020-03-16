#include "QA_Ferro.h"
#include <iostream>
#include <complex>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
using namespace std;

int main()
{
    //QA_Ferro(int Nt_half, int Ns_, double T_, double t_, double U_, double delmu_, double h_);
    int Nt_half = 16;
    int Ns = 1;
    double t = 0.;
    double T = 1.;
    double U = 5.;
    double delmutab[101];
    for (int i = 0; i < 101; i++)
        delmutab[i] = i * 0.2 - 10;
    double h = 1;
    ostringstream dir;
    dir << "../QAFdata/Ns" << Ns << "/T" << T << "/U" << U << "/";
    system(("mkdir -p " + dir.str()).c_str());
    ostringstream filename(dir.str(), ios::ate);
    filename << "Nt" << Nt_half * 2 << "_g(1,0)vsdelmu.dat";
    ofstream outfile(filename.str());
    if (!outfile)
    {
        cerr << "cannot open file \"" << filename.str() << "\"" << endl;
        exit(1);
    }
    QA_Ferro mysystem(Nt_half, Ns, T, t, U, delmutab[0], h);

    for (int delmuid = 0; delmuid < 101; delmuid++)
    {
        mysystem.setdelmu(delmutab[delmuid]);
        complex<double> gf = mysystem.greensfunction_up(0, 0);
        outfile << setiosflags(ios::fixed) << setprecision(1) << setw(5) << delmutab[delmuid] << "    "
                << resetiosflags(ios::floatfield) << setiosflags(ios::scientific) << setprecision(6)
                << setw(13) << gf.real() << "    " << setw(13) << gf.imag() << resetiosflags(ios::floatfield) << endl;
    }
    return 0;
}