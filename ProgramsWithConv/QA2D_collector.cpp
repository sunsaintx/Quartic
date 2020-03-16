#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <complex>
using namespace std;
using Complex = std::complex<double>;

int Nt_half = 128;
double T = 1.;
Complex *iw = new Complex[Nt_half];
Complex *g = new Complex[Nt_half];
double G(double tau);
int main()
{
    int Nx = 4;
    int Ny = 4;
    double U = 16;
    double delmu = 0;
    ostringstream infilename;
    infilename << "QAdata2D/"
               << "Nx" << Nx << "Ny" << Ny << "/T" << T
               << "/U" << U << "/delmu" << delmu << "/Nt" << Nt_half * 2
               << "/g_kx2ky0.dat";
    ifstream ginfile(infilename.str());
    double w, gr, gi;
    for (int n = 0; n < Nt_half; n++)
    {
        ginfile >> w >> gr >> gi;
        iw[n] = Complex(0, w);
        g[n] = Complex(gr, gi);
    }
    cout << G(0.5) << endl;
}

double G(double tau)
{
    static Complex I = Complex(0, 1);
    double ret;
    for (int n = 0; n < Nt_half; n++)
        ret += ((g[n] + 1. / iw[n]) * exp(iw[n] * tau)).real();
    return 2. * T * ret - 0.5;
}