CC=g++ -fopenmp -O3
CFLAGS = -m64 -I${MKLROOT}/include
LDFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

main: main.o QuarticApproximation2D.o
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

main.o:main.cpp QuarticApproximation2D.h
	$(CC) -c $<

QuarticApproximation2D.o:QuarticApproximation2D.cpp QuarticApproximation2D.h
	$(CC) $(CFLAGS) $(LDFLAGS) -c $<

.PHONY:clean
clean:
	rm -rf *.o main