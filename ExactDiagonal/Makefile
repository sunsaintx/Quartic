CC=g++ -fopenmp -O3
CFLAGS=-m64 -I${MKLROOT}/include
LDFLAGS=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lgomp -lpthread -lm -ldl

main: main.o ED.o
	$(CC) $^ ${LDFLAGS} ${CFLAGS} -o $@

main.o: main.cpp ED.h
	$(CC) -c $<

ED.o: ED.cpp ED.h
	$(CC) -c $(CFLAGS) $<

.PHONY:clean
clean:
	rm -rf *.o main
