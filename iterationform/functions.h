Complex ixp(Complex yy)
{
	return exp(I*yy);
}

inline int mt(int w0)
{
	return  (w0 + 2 * Nt) % Nt;
}

void print(const char str[]){
    time_t t=time(NULL);
    tm* tt=localtime(&t);
    cout<<put_time(tt,"%d-%m-%Y %H:%M:%S")<<":"<<str<<endl;
}

int vsum(int alpha,int beta,int gamma)//++-
{
	int alpha_k=alpha/Nt,alpha_n=alpha%Nt;
    int  beta_k= beta/Nt, beta_n= beta%Nt;
    int gamma_k=gamma/Nt, gamma_n=gamma%Nt;
    int res_k,res_n;
    res_k=(alpha_k+beta_k-gamma_k+Ns)%Ns;
    res_n=(alpha_n+beta_n-gamma_n+Nt)%Nt;
    return res_k*Nt+res_n;
}