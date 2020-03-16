#include <iostream>
#include <stdio.h>
#include <fstream>
#include <complex>
#include <ctime>
#include <cmath>
#include <iomanip>
using namespace std;
using Complex=complex<double>;

constexpr int Nt = 64;
constexpr int Ns = 4;
constexpr int Nts = Nt*Ns;
constexpr int NN = Nts*Nts;
constexpr double T = 1;
constexpr double U = 1;
constexpr double delmu = 0.;
constexpr double mu = U / 2. + delmu;
constexpr double Nit = 2000;

constexpr double pi = 3.1415926535897932385;
constexpr double f_5 = 0.5*U*T/Ns;

constexpr Complex I(0., 1.);
#include "functions.h"

int main(void){
    double p=0.3;
    print("starting...");
    Complex* iw=(Complex*)malloc(sizeof(Complex)*Nt);
    Complex* g=(Complex*)malloc(sizeof(Complex)*Nts);
    Complex* h=(Complex*)malloc(sizeof(Complex)*Nts);
    Complex* dd=(Complex*)malloc(sizeof(Complex)*Nts);
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
	}

    #pragma omp parallel for
    for (int alpha = 0; alpha < Nts; alpha++){
		int k=alpha/Nt, omega=alpha%Nt;
		g[alpha] = -1. / (iw[omega] - 2.*cos(2. * pi*k / Ns));
	}

    //initialize chains
    #pragma omp parallel for
    for(int n=0;n<NN;n++){
        c[n]=0.;
        d1[n]=0.;
        d2[n]=0.;
        r[n]=0.;
    }

    for(int it = 0;it < Nit;it++){
        print("start iteration..");
        cout<<"it = "<<it<<endl;
        Complex nh = 0.;
        for (int n = 0; n < Nts; n++){
			nh += T/Ns * g[n];
		}
        cout<<nh<<endl;

        #pragma omp parallel for
		for (int alpha = 0; alpha < Nts; alpha++){
            int k=alpha/Nt, omega=alpha%Nt;
			h[alpha] = -1. / (iw[omega] - 2.*cos(2. * pi*k / Ns)- delmu /*- U / 2.*/ + U * nh);
		}

        #pragma omp parallel for
        for(int beta=0;beta<Nts;beta++){
            for(int gamma=0;gamma<Nts;gamma++){
                //initialize new chains
                nc[Nts*beta+gamma]=0.;
                nd1[Nts*beta+gamma]=0.;
                nd2[Nts*beta+gamma]=0.;
                nr[Nts*beta+gamma]=0.;
                for(int nu=0;nu<Nts;nu++){
                    nc[Nts*beta+gamma]-=f_5*h[vsum(beta,gamma,nu)]*g[beta]*(d2[Nts*nu+gamma]-d1[Nts*nu+gamma]);
                    nc[Nts*beta+gamma]-=f_5*h[vsum(beta,gamma,nu)]*g[gamma]*(d2[Nts*nu+beta]-d1[Nts*nu+beta]);
                    nc[Nts*beta+gamma]-=4.*U*f_5*h[nu]*g[beta]*g[gamma]*g[vsum(beta,gamma,nu)];
                    //nd1[Nts*beta+gamma]+=f_5*h[nu]*(g[vsum(gamma,nu,beta)]+g[vsum(beta,nu,gamma)])*d1[Nts*beta+gamma];
                    nd1[Nts*beta+gamma]+=f_5*h[nu]*(g[vsum(gamma,nu,beta)]+g[vsum(beta,nu,gamma)])*d2[Nts*beta+gamma];
                    nd1[Nts*beta+gamma]-=f_5*h[vsum(beta,nu,gamma)]*g[beta]*(d1[Nts*nu+gamma]+d2[Nts*nu+gamma]);
                    nd1[Nts*beta+gamma]-=f_5*h[vsum(gamma,nu,beta)]*g[gamma]*(d1[Nts*beta+nu]+d2[Nts*beta+nu]);
                    nd2[Nts*beta+gamma]+=f_5*h[vsum(beta,nu,gamma)]*g[beta]*d1[Nts*nu+gamma];
                    nd2[Nts*beta+gamma]+=f_5*h[vsum(gamma,nu,beta)]*g[gamma]*d1[Nts*beta+nu];
                    nd2[Nts*beta+gamma]-=f_5*h[vsum(gamma,nu,beta)]*g[beta]*c[Nts*nu+gamma];
                    nd2[Nts*beta+gamma]-=f_5*h[vsum(beta,nu,gamma)]*g[gamma]*r[Nts*nu+beta];
                    nd2[Nts*beta+gamma]-=2.*U*f_5*h[nu]*g[beta]*g[gamma]*(g[vsum(beta,nu,gamma)]+g[vsum(gamma,nu,beta)]);
                    nr[Nts*beta+gamma]-=f_5*h[vsum(beta,gamma,nu)]*g[beta]*(d2[Nts*gamma+nu]-d1[Nts*gamma+nu]);
                    nr[Nts*beta+gamma]-=f_5*h[vsum(beta,gamma,nu)]*g[gamma]*(d2[Nts*beta+nu]-d1[Nts*beta+nu]);
                    nr[Nts*beta+gamma]-=4.*U*f_5*h[nu]*g[beta]*g[gamma]*g[vsum(beta,gamma,nu)];
                }
                //divisors
                Complex divc=1.,divd1=1.,divd2=1.,divr=1.;
                for(int nu=0;nu<Nts;nu++){
                    divc+=2.*f_5*h[nu]*g[vsum(beta,gamma,nu)];
                    divd1-=f_5*h[nu]*(g[vsum(gamma,nu,beta)]+g[vsum(beta,nu,gamma)]);
                    divd2+=f_5*h[nu]*(g[vsum(gamma,nu,beta)]+g[vsum(beta,nu,gamma)]);
                    divr+=2.*f_5*h[nu]*g[vsum(beta,gamma,nu)];
                }
                nc[Nts*beta+gamma]/=divc;
                nd1[Nts*beta+gamma]/=divd1;
                nd2[Nts*beta+gamma]/=divd2;
                nr[Nts*beta+gamma]/=divr;
            }
        }
        double normal=0.;
        for(int beta=0;beta<Nts;beta++){
            for(int gamma=0;gamma<Nts;gamma++){
                normal+=norm(nc[Nts*beta+gamma]-c[Nts*beta+gamma]);
                normal+=norm(nd1[Nts*beta+gamma]-d1[Nts*beta+gamma]);
                normal+=norm(nd2[Nts*beta+gamma]-d2[Nts*beta+gamma]);
                normal+=norm(nr[Nts*beta+gamma]-r[Nts*beta+gamma]);
            }
        }
        #pragma omp parallel for
        for(int n=0;n<NN;n++){
            c[n]=nc[n];
            d1[n]=nd1[n];
            d2[n]=nd2[n];
            r[n]=nr[n];
        }
        for(int n=0;n<Nts;n++){
            dd[n]=0;
            for(int nu=0;nu<Nts;nu++){
                dd[n]+=r[Nts*n+nu]+d1[Nts*n+nu]+2.*d2[Nts*n+nu];
            }
            dd[n]=(T/(6.*Ns))*dd[n];
        }

        //get new g[n]
        for (int alpha = 0; alpha < Nts; alpha++){
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
    

    for(int omega_n=0;omega_n<10;omega_n++){
        //cout << "n = " << omega_n << endl;
        int idx = 1;
        Complex chi = 0;
        for(int gamma_n = 0; gamma_n < Nt; gamma_n++)
        {
            for(int gamma_k = 0; gamma_k < Ns; gamma_k++)
            {
                int gamma = gamma_k * Nt + gamma_n;
                for(int beta_k = 0; beta_k < Ns; beta_k++)
                {
                    int beta = beta_k * Nt + (gamma_n + omega_n) % Nt;
                    chi += exp(2. * I * pi * double(idx) * double(beta_k - gamma_k) / double(Ns)) * (d2[Nts * beta + gamma] / U + 2. * g[beta] * g[gamma]);
                }
            }
        }
        cout << omega_n << " " << (-chi * T /double(Ns * Ns)).real() << endl;
    }
    /* char filename[100];
    for(int k=0;k<Ns;k++){
        sprintf(filename,"./pseudo_gap/U4/k%d.dat",k);
        ofstream outfile(filename); */
        /*for(int n=Nt/2;n<Nt;n++){
		    outfile<<iw[n].imag()<<"	"<<-g[k*Nt+n].real()
			    <<"    "<<-g[k*Nt+n].imag()<<endl;
	    }*/
    /*     for(int n=0;n<Nt/2;n++){
		    outfile<<iw[n].imag()<<"	"<<-g[k*Nt+n].real()
			    <<"    "<<-g[k*Nt+n].imag()<<endl;
	    }
        outfile.close();
    } */
    return 0;
}
