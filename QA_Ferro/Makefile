CC=g++ -fopenmp -O3

main: main.o QA_Ferro.o
	$(CC) $^ -o $@

main.o:main.cpp QA_Ferro.h
	$(CC) -c $<

QA_Ferro.o:QA_Ferro.cpp QA_Ferro.h
	$(CC) -c $<