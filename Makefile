# @ref https://www.gnu.org/software/libc/manual/html_node/Feature-Test-Macros.html#index-_005fFILE_005fOFFSET_005fBITS
# @ref https://man7.org/linux/man-pages/man3/fseeko.3.html
# @ref https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# @ref https://simonbyrne.github.io/notes/fastmath/
# @note Compiling with OpenMP enables multithreaded execution.
# OpenMP defaults to using all available threads.
# This behavior can be modified by using the env var OMP_NUM_THREADS at runtime:
# 	OMP_NUM_THREADS=4 ./qwen model/weights.bin

CC = gcc
SRC = examples/qwen.c
BIN = qwen
OPTIONS = -lm -fopenmp -D_FILE_OFFSET_BITS=64

.PHONY: debug
debug: $(SRC)
	$(CC) $(SRC) $(OPTIONS) -g3 -fsanitize=address,undefined -fno-omit-frame-pointer -o $(BIN)

.PHONY: release
release: $(SRC)
	$(CC) $(SRC) $(OPTIONS) -O3 -o $(BIN)

.PHONY: fast
openmp: $(SRC)
	$(CC) $(SRC) $(OPTIONS) -Ofast -march=native -o $(BIN)

.PHONY: clean
clean:
	rm -f $(BIN)
