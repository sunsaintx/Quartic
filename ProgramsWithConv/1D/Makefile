CC=g++ -fopenmp -O3
CFLAGS = -m64 -I${MKLROOT}/include
LDFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

main: main.o QuarticApproximation.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

main.o:main.cpp QuarticApproximation.h
	$(CC) -c $<

QuarticApproximation.o:QuarticApproximation.cpp QuarticApproximation.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c $<

.PHONY:clean
clean:
	rm -rf *.o main