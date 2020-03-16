Complex ixp(Complex yy)
{
	return exp(I*yy);
}

size_t mt(size_t w0)
{
	return  (w0 + 2 * Nt) % Nt;
}

MKL_Complex16* StdToMkl(Complex* complexPtr)
{
	return reinterpret_cast<MKL_Complex16*>(complexPtr);
}

void print(const char str[]){
    time_t t=time(NULL);
    tm* tt=localtime(&t);
    cout<<put_time(tt,"%d-%m-%Y %H:%M:%S")<<":"<<str<<endl;
}