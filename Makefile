# @ref https://www.gnu.org/software/libc/manual/html_node/Feature-Test-Macros.html#index-_005fFILE_005fOFFSET_005fBITS
# @ref https://man7.org/linux/man-pages/man3/fseeko.3.html
# @ref https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# @ref https://simonbyrne.github.io/notes/fastmath/
# @note Compiling with OpenMP enables multithreaded execution.
# OpenMP defaults to using all available threads.
# This behavior can be modified by using the env var OMP_NUM_THREADS at runtime:
# 	OMP_NUM_THREADS=4 ./qwen model/weights.bin

CC = gcc

OPTIONS = -lm -fopenmp -D_FILE_OFFSET_BITS=64

RUNQ_SRC = examples/runq.c
RUNQ_BIN = runq

QWEN_SRC = examples/qwen.c
QWEN_BIN = qwen

.PHONY: debug
debug: $(QWEN_SRC)
	$(CC) $(QWEN_SRC) $(OPTIONS) -g3 -fsanitize=address,undefined -fno-omit-frame-pointer -o $(QWEN_BIN)

.PHONY: release
release: $(QWEN_SRC)
	$(CC) $(QWEN_SRC) $(OPTIONS) -O3 -o $(QWEN_BIN)

.PHONY: optimized
openmp: $(QWEN_SRC)
	$(CC) $(QWEN_SRC) $(OPTIONS) -Ofast -march=native -o $(QWEN_BIN)

.PHONY: runq
runq: examples/runq.c
	$(CC) examples/runq.c $(OPTIONS) -Ofast -march=native -o $(QWEN_BIN)

.PHONY: clean
clean:
	rm -f $(RUNQ_BIN) $(QWEN_BIN)
