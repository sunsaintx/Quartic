#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <complex>
#include <cmath>
#include "mkl.h"
#include "mkl_spblas.h"
using namespace std;
using Complex = complex<double>;

constexpr int Nt = 32;
constexpr int NN = Nt*Nt, NNN = Nt*NN;
constexpr int DIM = 4 * NN;

constexpr double T = 1.;
constexpr double U = 16.;
constexpr double delmu = 0.;
constexpr double mu = U / 2. + delmu;
constexpr double Nit = 500;

constexpr double pi = 3.1415926535897932385;
constexpr double f_5 = 0.5*U*T;

constexpr Complex I(0., 1.);

//functions declaration
#include "functions.h"

int main(void){
	double p=0.05;
    Complex iw[Nt], g[Nt], h[Nt], dd[Nt];
    print("starting...");
    cout << "dim=" << DIM << endl;
    
	/*#pragma omp parallel for
    for(int n = 0; n < Nt; n++){
        iw[n] = Nt * T*(ixp(pi*(2.*n + 1.) / Nt) - 1.);
        g[n] = -1. / (iw[n]);//Is initial for iterations
    }*/

	#pragma omp parallel for
	for(int n = 0; n < Nt; n++){
		if(n<Nt/2)
			iw[n]=I*pi*T*(2.*n+1.);
		else
			iw[n]=I*pi*T*(2.*(n-Nt)+1);
		g[n] = -1. / (iw[n]);
	}

	const int nnz1 = 9 * NNN, nnz2 = 5 * NN, nnz3 = 7 * NNN, nnz4 = nnz1 + nnz2, nnz = nnz4 + nnz3;
	int NNZ1 = nnz1, NNZ2 = nnz2, NNZ3 = nnz3;
	Complex *matcoo1=(Complex*)malloc(sizeof(Complex)*nnz1);
	Complex *matcoo2=(Complex*)malloc(sizeof(Complex)*nnz2);
	Complex *matcoo3=(Complex*)malloc(sizeof(Complex)*nnz3);
	int *imatcoo1=(int*)malloc(sizeof(int)*nnz1);
	int *imatcoo2=(int*)malloc(sizeof(int)*nnz2);
	int *imatcoo3=(int*)malloc(sizeof(int)*nnz3);
    int *jmatcoo1=(int*)malloc(sizeof(int)*nnz1);
    int *jmatcoo2=(int*)malloc(sizeof(int)*nnz2);
    int *jmatcoo3=(int*)malloc(sizeof(int)*nnz3);
	Complex *vec = (Complex *)malloc(sizeof(Complex) * DIM);
	Complex *xpar= (Complex *)malloc(sizeof(Complex) * DIM);
	//start iteration
	for (int it = 0; it < Nit; it++){
		cout<<"it = "<<it+1<<endl;
		Complex nh = 0.;
		for (int n = 0; n < Nt; n++){
			nh += T * g[n];
		}
		#pragma omp parallel for
		for (int n = 0; n < Nt; n++){
			h[n] = -1. / (iw[n] - delmu /*- U / 2*/ + U * nh);
		}
		print("Allocating sparse matrix...");
		#pragma omp parallel for
		for (int b = 0; b < Nt; b++)
		{
			for (int c = 0; c < Nt; c++)
			{
				for (int lam = 0; lam < Nt; lam++)
				{
					matcoo1[b + Nt * c + NN * lam] = -f_5 * h[mt(b + c - lam)] * g[b];
					imatcoo1[b + Nt * c + NN * lam] = b + Nt * c + 1;
					jmatcoo1[b + Nt * c + NN * lam] = NN + lam + Nt * c + 1;

					matcoo1[NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b + c - lam)] * g[b];
					imatcoo1[NNN + b + Nt * c + NN * lam] = b + Nt * c + 1;
					jmatcoo1[NNN + b + Nt * c + NN * lam] = 2 * NN + lam + Nt * c + 1;

					matcoo1[2 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b - c + lam)] * g[b];
					imatcoo1[2 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * c + 1;
					jmatcoo1[2 * NNN + b + Nt * c + NN * lam] = NN + lam + Nt * c + 1;

					matcoo1[3 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b - c + lam)] * g[b];
					imatcoo1[3 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * c + 1;
					jmatcoo1[3 * NNN + b + Nt * c + NN * lam] = 2 * NN + lam + Nt * c + 1;

					matcoo1[4 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(c - b + lam)] * g[b];
					imatcoo1[4 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * c + 1;
					jmatcoo1[4 * NNN + b + Nt * c + NN * lam] = lam + Nt * c + 1;

					matcoo1[5 * NNN + b + Nt * c + NN * lam] = -f_5 * h[mt(b - c + lam)] * g[b];
					imatcoo1[5 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * c + 1;
					jmatcoo1[5 * NNN + b + Nt * c + NN * lam] = NN + lam + Nt * c + 1;

					matcoo1[6 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b - c + lam)] * g[c];
					imatcoo1[6 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * c + 1;
					jmatcoo1[6 * NNN + b + Nt * c + NN * lam] = 3 * NN + lam + Nt * b + 1;

					matcoo1[7 * NNN + b + Nt * c + NN * lam] = -f_5 * h[mt(b + c - lam)] * g[b];
					imatcoo1[7 * NNN + b + Nt * c + NN * lam] = 3 * NN + b + Nt * c + 1;
					jmatcoo1[7 * NNN + b + Nt * c + NN * lam] = NN + c + Nt * lam + 1;

					matcoo1[8 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b + c - lam)] * g[b];
					imatcoo1[8 * NNN + b + Nt * c + NN * lam] = 3 * NN + b + Nt * c + 1;
					jmatcoo1[8 * NNN + b + Nt * c + NN * lam] = 2 * NN + c + Nt * lam + 1;
				}
			}
		}
		#pragma omp parallel for
		for (int b = 0; b < Nt; b++)
		{
			for (int c = 0; c < Nt; c++)
			{
				Complex sum1 = 0. , sum2 = 0.;
				for (int nu = 0; nu < Nt; nu++)
				{
					sum1 += h[nu] * g[mt(b + c - nu)];
					sum2 += h[nu] * (g[mt(c - b + nu)] + g[mt(b - c + nu)]);
				}
				matcoo2[b + Nt * c] = 1. + 2.*f_5*sum1;
				imatcoo2[b + Nt * c] = b + Nt * c + 1;
				jmatcoo2[b + Nt * c] = b + Nt * c + 1;

				matcoo2[NN + b + Nt * c] = 1. - f_5 * sum2;
				imatcoo2[NN + b + Nt * c] = NN + b + Nt * c + 1;
				jmatcoo2[NN + b + Nt * c] = NN + b + Nt * c + 1;

				matcoo2[2 * NN + b + Nt * c] = -f_5 * sum2;
				imatcoo2[2 * NN + b + Nt * c] = NN + b + Nt * c + 1;
				jmatcoo2[2 * NN + b + Nt * c] = 2 * NN + b + Nt * c + 1;

				matcoo2[3 * NN + b + Nt * c] = 1. + f_5 * sum2;
				imatcoo2[3 * NN + b + Nt * c] = 2 * NN + b + Nt * c + 1;
				jmatcoo2[3 * NN + b + Nt * c] = 2 * NN + b + Nt * c + 1;

				matcoo2[4 * NN + b + Nt * c] = 1. + 2.*f_5*sum1;
				imatcoo2[4 * NN + b + Nt * c] = 3 * NN + b + Nt * c + 1;
				jmatcoo2[4 * NN + b + Nt * c] = 3 * NN + b + Nt * c + 1;
			}
		}
		#pragma omp parallel for
		for (int b = 0; b < Nt; b++)
		{
			for (int c = 0; c < Nt; c++)
			{
				for (int lam = 0; lam < Nt; lam++)
				{
					matcoo3[b + Nt * c + NN * lam] = -f_5 * h[mt(b + c - lam)] * g[c];
					imatcoo3[b + Nt * c + NN * lam] = b + Nt * c + 1;
					jmatcoo3[b + Nt * c + NN * lam] = NN + lam + Nt * b + 1;

					matcoo3[NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b + c - lam)] * g[c];
					imatcoo3[NNN + b + Nt * c + NN * lam] = b + Nt * c + 1;
					jmatcoo3[NNN + b + Nt * c + NN * lam] = 2 * NN + lam + Nt * b + 1;

					matcoo3[2 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(c - b + lam)] * g[c];
					imatcoo3[2 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * c + 1;
					jmatcoo3[2 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * lam + 1;

					matcoo3[3 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(c - b + lam)] * g[c];
					imatcoo3[3 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * c + 1;
					jmatcoo3[3 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * lam + 1;

					matcoo3[4 * NNN + b + Nt * c + NN * lam] = -f_5 * h[mt(c - b + lam)] * g[c];
					imatcoo3[4 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * c + 1;
					jmatcoo3[4 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * lam + 1;

					matcoo3[5 * NNN + b + Nt * c + NN * lam] = -f_5 * h[mt(b + c - lam)] * g[c];
					imatcoo3[5 * NNN + b + Nt * c + NN * lam] = 3 * NN + b + Nt * c + 1;
					jmatcoo3[5 * NNN + b + Nt * c + NN * lam] = NN + b + Nt * lam + 1;

					matcoo3[6 * NNN + b + Nt * c + NN * lam] = f_5 * h[mt(b + c - lam)] * g[c];
					imatcoo3[6 * NNN + b + Nt * c + NN * lam] = 3 * NN + b + Nt * c + 1;
					jmatcoo3[6 * NNN + b + Nt * c + NN * lam] = 2 * NN + b + Nt * lam + 1;
				}
			}
		}
		//creat coo matrix handle
		sparse_status_t status;
		sparse_index_base_t indexing = SPARSE_INDEX_BASE_ONE;
		sparse_matrix_t Acoo, Bcoo, Ccoo;
		status = mkl_sparse_z_create_coo(&Acoo, indexing, DIM, DIM, NNZ1, imatcoo1, jmatcoo1, StdToMkl(matcoo1));
		if (status != 0)
		{
			cerr << " mkl_sparse_z_create_coo Status: " << status << endl;
			return status;
		}
		status = mkl_sparse_z_create_coo(&Bcoo, indexing, DIM, DIM, NNZ2, imatcoo2, jmatcoo2, StdToMkl(matcoo2));
		if (status != 0)
		{
			cerr << " mkl_sparse_z_create_coo Status: " << status << endl;
			return status;
		}
		status = mkl_sparse_z_create_coo(&Ccoo, indexing, DIM, DIM, NNZ3, imatcoo3, jmatcoo3, StdToMkl(matcoo3));
		if (status != 0)
		{
			cerr << " mkl_sparse_z_create_coo Status: " << status << endl;
			return status;
		}
		//convert to csr
		sparse_matrix_t Mat1, Mat2, Mat3;
		sparse_operation_t operation = SPARSE_OPERATION_NON_TRANSPOSE;
		status = mkl_sparse_convert_csr(Acoo, operation, &Mat1);
		if (status != 0)
		{
			cerr << " mkl_sparse_convert Status: " << status << endl;
			return status;
		}
		mkl_sparse_destroy(Acoo);
		status = mkl_sparse_convert_csr(Bcoo, operation, &Mat2);
		if (status != 0)
		{
			cerr << " mkl_sparse_convert Status: " << status << endl;
			return status;
		}
		mkl_sparse_destroy(Bcoo);
		status = mkl_sparse_convert_csr(Ccoo, operation, &Mat3);
		if (status != 0)
		{
			cerr << " mkl_sparse_convert Status: " << status << endl;
			return status;
		}
		mkl_sparse_destroy(Ccoo);
		//summing up sparse matrices
		sparse_matrix_t Mat12, Mat;
		MKL_Complex16 alpha={1,0};
		status = mkl_sparse_z_add(operation, Mat1, alpha, Mat2, &Mat12);
		if (status != 0)
		{
			cerr << " mkl_sparse_z_add Status: " << status << endl;
			return status;
		}
		mkl_sparse_destroy(Mat1);
		mkl_sparse_destroy(Mat2);
		status = mkl_sparse_z_add(operation, Mat12, alpha, Mat3, &Mat);
		if (status != 0)
		{
			cerr << " mkl_sparse_z_add Status: " << status << endl;
			return status;
		}
		mkl_sparse_destroy(Mat12);
		mkl_sparse_destroy(Mat3);
		//export the matrix in csr form
		sparse_index_base_t index;
		int rows,cols;
		int* imat,*jmat,*kmat;
		MKL_Complex16 *mat;
		status = mkl_sparse_z_export_csr(Mat, &index, &rows, &cols, &imat, &kmat, &jmat, &mat);
		if (status != 0)
		{
			cerr << "mkl_sparse_z_export_csr Status: " << status << endl;
			return status;
		}
		if(rows!=DIM||cols!=DIM)
			cout<<"the exported matrix has a wrong dimension"<<endl;
		//for(int i=0;i<=dim;i++) cout<<imat[i]<<"	"<<kmat[i]<<endl;
		/*if(index==SPARSE_INDEX_BASE_ONE) cout<<"The matrix is base one"<<endl;
		else if(index==SPARSE_INDEX_BASE_ZERO) cout<<"The matrix is base zero"<<endl;*/
		//mkl_sparse_destroy(Mat);
		#pragma omp parallel for
		for (int b = 0; b < Nt; b++)
		{
			for (int c = 0; c < Nt; c++)
			{
				Complex sum1 = 0., sum2 = 0.;
				for (int nu = 0; nu < Nt; nu++)
				{
					sum1 += h[nu] * g[mt(b + c - nu)];
					sum2 += h[nu] * (g[mt(b - c + nu)] + g[mt(c - b + nu)]);
				}
				vec[b + Nt * c] = -4.*f_5*U*g[b] * g[c] * sum1;
				vec[NN + b + Nt * c] = 0.;
				vec[2 * NN + b + Nt * c] = -2.*f_5*U*g[b] * g[c] * sum2;
				vec[3 * NN + b + Nt * c] = -4.*f_5*U*g[b] * g[c] * sum1;
			}
		}
		//start pardiso
		int mtype = 13;//Complex
		int nrhs = 1; /* Number of right hand sides. */
		void *pt[64];  /* void *pt[64] should be OK on both architectures */
		int iparm[64];
		int maxfct, mnum, phase, error, msglvl;
		Complex ddum; /* Double dummy */
		int idum; /* Integer dummy. */

		for (int i = 0; i < 64; i++)
		{
			iparm[i] = 0;
		}
		iparm[0] = 1; /* No solver default */
		iparm[1] = 2; /* Fill-in reordering from METIS */
		iparm[2] = 1; /* Numbers of processors, value of OMP_NUM_THREADS */
		iparm[3] = 0; /* No iterative-direct algorithm */
		iparm[4] = 0; /* No user fill-in reducing permutation */
		iparm[5] = 1; /* Write solution into x for 0, write into b for 1*/
		iparm[7] = 2; /* Max numbers of iterative refinement steps */
		iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
		iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
		iparm[13] = 0; /* Output: Number of perturbed pivots */
		iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
		iparm[18] = -1; /* Output: Mflops for LU factorization */
		iparm[19] = 0; /* Output: Numbers of CG Iterations */
		if(index==SPARSE_INDEX_BASE_ZERO) 	iparm[34] = 1; /*Zero base*/
		maxfct = 1; /* Maximum number of numerical factorizations. */
		mnum = 1; /* Which factorization to use. */
		msglvl = 0; /*1  Print statistical information in file */
		error = 0; /* Initialize error flag */

		for (int i = 0; i < 64; i++)
		{
			pt[i] = 0;
		}
		print("Start pardiso...");
		//Phase 1. Reordering and Symbolic Factorization. Allocates memory necessary for the factorization.
		phase = 11;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &DIM, mat, imat, jmat, &idum, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);
		if (error != 0)
		{
			printf("\nERROR during symbolic factorization: %d", error);
			exit(1);
		}
		//Phase 2. Numerical factorization. 
		phase = 22;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &DIM, mat, imat, jmat, &idum, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);
		if (error != 0)
		{
			printf("\nERROR during numerical factorization: %d", error);
			exit(2);
		}
		//Phase 3. Back substitution and iterative refinement.
		phase = 33;
		iparm[7] = 2; //Max numbers of iterative refinement steps. Set right hand side to one. 
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &DIM, mat, imat, jmat, &idum, &nrhs,
			iparm, &msglvl, StdToMkl(vec), StdToMkl(xpar), &error);
		if (error != 0)
		{
			printf("\nERROR during solution: %d", error);
			exit(3);
		}
		//4. Phase (-1). Termination and release of memory.
		phase = -1;
		pardiso(pt, &maxfct, &mnum, &mtype, &phase, &DIM, &ddum, imat, jmat, &idum, &nrhs,
			iparm, &msglvl, &ddum, &ddum, &error);
		print("Finish pardiso");
		for (int n = 0; n < Nt; n++)
		{
			dd[n] = 0.;
			for (int nu = 0; nu < Nt; nu++)
			{
				dd[n] += (T / 6.)*(vec[NN + n + Nt * nu] + 2.*vec[2 * NN + n + Nt * nu] + vec[3 * NN + n + Nt * nu]);
			}
		}
		
		/*#pragma omp parallel for
		for (int n = 0; n<Nt; n++)
		{
			g[n] = h[n] * (1. + dd[n]);
		}*/

		#pragma omp parallel for
		for (int alpha = 0; alpha < Nt; alpha++)
		{
		//dd[alpha] = h[alpha]*(1.+dd[alpha]);
		dd[alpha] = h[alpha] / (1. - h[alpha] * dd[alpha] / g[alpha]);
		g[alpha] = p * dd[alpha] + (1.-p) * g[alpha];
		}
		cout << "gnew[0]=" << g[0].real() << "+I*(" << g[0].imag() << ")" << endl;
		mkl_sparse_destroy(Mat);
	}
	ofstream outfile("T1U8Nt64.dat");
	for(int n=Nt/2;n<Nt;n++){
		outfile<<iw[n].imag()<<"	"<<-g[n].real()
			<<"    "<<-g[n].imag()<<endl;
	}
	for(int n=0;n<Nt/2;n++){
		outfile<<iw[n].imag()<<"	"<<-g[n].real()
			<<"    "<<-g[n].imag()<<endl;
	}
	outfile.close();
    return 0;
}
