#!/bin/sh

set -xe

CFLAGS="-Wall -Wextra -ggdb `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm"

clang $CFLAGS -o ./seam ./*.c $LIBS -L./bin/

./seam $1
