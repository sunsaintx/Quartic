CC=g++ -fopenmp
CFLAGS=-m64 -I${MKLROOT}/include
LDFLAGS=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lgomp -lpthread -lm -ldl

.PHONY: default clean

default:QA0D.cpp
	${CC} -o main $< ${LDFLAGS} ${CFLAGS}

clean:
	rm -rf *.o main
