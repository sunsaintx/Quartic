#include <iostream>
#include <complex>
#include <ctime>
#include <cmath>
#include <iomanip>
using namespace std;
using Complex=complex<double>;

constexpr int Nt = 128;
constexpr int NN = Nt*Nt;
constexpr double T = 1.;
constexpr double U = 16.;
constexpr double delmu = 0.;
constexpr double mu = U / 2. + delmu;
constexpr double Nit = 2000;

constexpr double pi = 3.1415926535897932385;
constexpr double f_5 = 0.5*U*T;

constexpr Complex I(0., 1.);
#include "functions.h"

int main(void){
    double p=0.15;
    print("starting...");
    Complex* iw=(Complex*)malloc(sizeof(Complex)*Nt);
    Complex* g=(Complex*)malloc(sizeof(Complex)*Nt);
    Complex* h=(Complex*)malloc(sizeof(Complex)*Nt);
    Complex* dd=(Complex*)malloc(sizeof(Complex)*Nt);
    Complex* c=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* d1=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* d2=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* r=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* nc=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* nd1=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* nd2=(Complex*)malloc(sizeof(Complex)*NN);
    Complex* nr=(Complex*)malloc(sizeof(Complex)*NN);

    #pragma omp parallel for
	for(int n = 0; n < Nt; n++){
		if(n<Nt/2)
			iw[n]=I*pi*T*(2.*n+1.);
		else
			iw[n]=I*pi*T*(2.*(n-Nt)+1.);
		g[n] = -1. / (iw[n]);
	}

    //initialize chains
    for(int beta=0;beta<Nt;beta++){
        for(int gamma=0;gamma<Nt;gamma++){
            c[Nt*beta+gamma]=0.;
            d1[Nt*beta+gamma]=0.;
            d2[Nt*beta+gamma]=0.;
            r[Nt*beta+gamma]=0.;
        }
    }

    for(int it = 0;it < Nit;it++){
        print("start iteration..");
        cout<<"it = "<<it<<endl;
        Complex nh = 0.;
		for (int n = 0; n < Nt; n++){
			nh += T * g[n];
		}
        //nh=0.;
        //cout<<"nh="<<nh<<endl;

		#pragma omp parallel for
		for (int n = 0; n < Nt; n++){
			h[n] = -1. / (iw[n] - delmu /*- U / 2.*/ + U * nh);
		}

        #pragma omp parallel for
        for(int beta=0;beta<Nt;beta++){
            for(int gamma=0;gamma<Nt;gamma++){
                //initialize new chains
                nc[Nt*beta+gamma]=0.;
                nd1[Nt*beta+gamma]=0.;
                nd2[Nt*beta+gamma]=0.;
                nr[Nt*beta+gamma]=0.;
                for(int nu=0;nu<Nt;nu++){
                    nc[Nt*beta+gamma]-=f_5*h[mt(beta+gamma-nu)]*g[beta]*(d2[Nt*nu+gamma]-d1[Nt*nu+gamma]);
                    nc[Nt*beta+gamma]-=f_5*h[mt(beta+gamma-nu)]*g[gamma]*(d2[Nt*nu+beta]-d1[Nt*nu+beta]);
                    nc[Nt*beta+gamma]-=4.*U*f_5*h[nu]*g[beta]*g[gamma]*g[mt(beta+gamma-nu)];
                    //nd1[Nt*beta+gamma]+=f_5*h[nu]*(g[mt(gamma-beta+nu)]+g[mt(beta-gamma+nu)])*d1[Nt*beta+gamma];
                    nd1[Nt*beta+gamma]+=f_5*h[nu]*(g[mt(gamma-beta+nu)]+g[mt(beta-gamma+nu)])*d2[Nt*beta+gamma];
                    nd1[Nt*beta+gamma]-=f_5*h[mt(beta-gamma+nu)]*g[beta]*(d1[Nt*nu+gamma]+d2[Nt*nu+gamma]);
                    nd1[Nt*beta+gamma]-=f_5*h[mt(gamma-beta+nu)]*g[gamma]*(d1[Nt*beta+nu]+d2[Nt*beta+nu]);
                    nd2[Nt*beta+gamma]+=f_5*h[mt(beta-gamma+nu)]*g[beta]*d1[Nt*nu+gamma];
                    nd2[Nt*beta+gamma]+=f_5*h[mt(gamma-beta+nu)]*g[gamma]*d1[Nt*beta+nu];
                    nd2[Nt*beta+gamma]-=f_5*h[mt(gamma-beta+nu)]*g[beta]*c[Nt*nu+gamma];
                    nd2[Nt*beta+gamma]-=f_5*h[mt(beta-gamma+nu)]*g[gamma]*r[Nt*nu+beta];
                    nd2[Nt*beta+gamma]-=2.*U*f_5*h[nu]*g[beta]*g[gamma]*(g[mt(beta-gamma+nu)]+g[mt(gamma-beta+nu)]);
                    nr[Nt*beta+gamma]-=f_5*h[mt(beta+gamma-nu)]*g[beta]*(d2[Nt*gamma+nu]-d1[Nt*gamma+nu]);
                    nr[Nt*beta+gamma]-=f_5*h[mt(beta+gamma-nu)]*g[gamma]*(d2[Nt*beta+nu]-d1[Nt*beta+nu]);
                    nr[Nt*beta+gamma]-=4.*U*f_5*h[nu]*g[beta]*g[gamma]*g[mt(beta+gamma-nu)];
                }
                //divisors
                Complex divc=1.,divd1=1.,divd2=1.,divr=1.;
                for(int nu=0;nu<Nt;nu++){
                    divc+=2.*f_5*h[nu]*g[mt(beta+gamma-nu)];
                    divd1-=f_5*h[nu]*(g[mt(gamma-beta+nu)]+g[mt(beta-gamma+nu)]);
                    divd2+=f_5*h[nu]*(g[mt(nu+gamma-beta)]+g[mt(beta+nu-gamma)]);
                    divr+=2.*f_5*h[nu]*g[mt(beta+gamma-nu)];
                }
                nc[Nt*beta+gamma]/=divc;
                nd1[Nt*beta+gamma]/=divd1;
                nd2[Nt*beta+gamma]/=divd2;
                nr[Nt*beta+gamma]/=divr;
            }
        }

        double normal=0.;
        for(int beta=0;beta<Nt;beta++){
            for(int gamma=0;gamma<Nt;gamma++){
                normal+=norm(nc[Nt*beta+gamma]-c[Nt*beta+gamma]);
                normal+=norm(nd1[Nt*beta+gamma]-d1[Nt*beta+gamma]);
                normal+=norm(nd2[Nt*beta+gamma]-d2[Nt*beta+gamma]);
                normal+=norm(nr[Nt*beta+gamma]-r[Nt*beta+gamma]);
            }
        }

        //cout<<"nc[0]-c[0] = "<<nc[0]-c[0]<<endl;

        //cout<<normal<<endl;
        #pragma omp parallel for
        for(int beta=0;beta<Nt;beta++){
            for(int gamma=0;gamma<Nt;gamma++){
                c[Nt*beta+gamma]=nc[Nt*beta+gamma];
                d1[Nt*beta+gamma]=nd1[Nt*beta+gamma];
                d2[Nt*beta+gamma]=nd2[Nt*beta+gamma];
                r[Nt*beta+gamma]=nr[Nt*beta+gamma];
            }
        }

        //calculate dd[n]
        for(int n=0;n<Nt;n++){
            dd[n]=0;
            for(int nu=0;nu<Nt;nu++){
                dd[n]+=r[Nt*n+nu]+d1[Nt*n+nu]+2.*d2[Nt*n+nu];
            }
            dd[n]=(T/6.)*dd[n];
        }
        
        //get new g[n]
        for (int alpha = 0; alpha < Nt; alpha++){
		    //g[alpha] = h[alpha]*(1.+dd[alpha]);
	        dd[alpha] = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
		    g[alpha] = p * dd[alpha] + (1.-p) * g[alpha];
		}
        cout<<"g[0] = "<<g[0].real()<<"+I*"<<g[0].imag()<<endl;
        print("finish iteration");
        if(normal<0.00000000001) break;
    }
    cout<<"c[0] = "<<c[0]<<"    "
        <<"r[0] = "<<r[0]<<endl;
    return 0;
}
