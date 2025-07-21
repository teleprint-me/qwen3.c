# Root
cmake_minimum_required(VERSION 3.17)
project("qwen3" LANGUAGES C)

set(CMAKE_C_STANDARD 17)

# Required for clangd support in Zeditor (Zed Editor)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WARN "-Wall -Wextra -Wpedantic -Werror -Wformat-security -Wshadow -fexceptions")
set(EXTRA_WARN "-Wformat -Wnull-dereference -Wdouble-promotion")

set(SANITIZE "-fsanitize=address,undefined -fno-omit-frame-pointer")
set(ANALYSIS "-Wanalyzer-double-free -Wanalyzer-file-leak -Wanalyzer-malloc-leak -Wanalyzer-null-dereference -Wanalyzer-out-of-bounds -Wanalyzer-va-list-leak")

set(DEBUG "-D_FILE_OFFSET_BITS=64 -g3")
set(RELEASE "-D_FILE_OFFSET_BITS=64 -Ofast -march=native")

find_package(OpenMP REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${WARN} ${EXTRA_WARN} ${SANITIZE} ${ANALYSIS} ${DEBUG}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${WARN} ${RELEASE}")
endif()

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

add_library(qwen3 SHARED
    "src/tokenizer.c"
    "src/q8.c"
    "src/model.c"
    "src/forward.c"
    "src/xorshift.c"
    "src/sampler.c"
)
target_include_directories("qwen3" PUBLIC include)
target_link_libraries("qwen3" PUBLIC m)
add_subdirectory(examples)
