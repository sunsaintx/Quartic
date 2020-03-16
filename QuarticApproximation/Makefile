CC=g++ -fopenmp -O3

main: main.o QuarticApproximation.o
	$(CC) $^ -o $@

main.o:main.cpp QuarticApproximation.h
	$(CC) -c $<

QuarticApproximation.o:QuarticApproximation.cpp QuarticApproximation.h
	$(CC) -c $<