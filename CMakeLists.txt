# Root
cmake_minimum_required(VERSION 3.17)
project("qwen3" LANGUAGES C)

set(CMAKE_C_STANDARD 17)

# Required for clangd support in Zeditor (Zed Editor)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(COMMON_FLAGS "-D_FILE_OFFSET_BITS=64 -Wall -Wextra -Wpedantic -Werror -Wformat-security -Wshadow -fexceptions")
set(DEBUG_SANITIZERS "-fsanitize=address,undefined -fno-omit-frame-pointer")
set(DEBUG_EXTRA_WARNINGS "-Wformat -Wnull-dereference -Wdouble-promotion")
set(DEBUG_ANALYSIS "-Wanalyzer-double-free -Wanalyzer-file-leak -Wanalyzer-malloc-leak -Wanalyzer-null-dereference -Wanalyzer-out-of-bounds -Wanalyzer-va-list-leak")
set(DEBUG_FLAGS "-g3")
set(RELEASE_FLAGS "-Ofast -march=native")

find_package(OpenMP REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} ${OpenMP_C_FLAGS} ${DEBUG_SANITIZERS} ${DEBUG_EXTRA_WARNINGS} ${DEBUG_ANALYSIS} ${DEBUG_FLAGS}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} ${OpenMP_C_FLAGS} ${RELEASE_FLAGS}")
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
