CC ?= gcc
CFLAGS ?= -std=c99 -Wall -Wextra -Werror -pedantic -O2 -fPIC -Iinclude
UNAME_S := $(shell uname -s)

SRCS = src/cusum.c src/kalman_chisq.c src/water_filling_scheduler.c src/secoc.c
OBJS = $(patsubst src/%.c,build/%.o,$(SRCS))

STATIC_LIB = build/libseccharge.a
ifeq ($(UNAME_S),Darwin)
    SHARED_LIB = build/libseccharge.dylib
else
    SHARED_LIB = build/libseccharge.so
endif

TEST_BIN = build/test_runner

.PHONY: all clean test

all: $(STATIC_LIB) $(SHARED_LIB) $(TEST_BIN)

build:
	mkdir -p build

build/%.o: src/%.c | build
	$(CC) $(CFLAGS) -c $< -o $@

$(STATIC_LIB): $(OBJS)
	ar rcs $@ $^

$(SHARED_LIB): $(OBJS)
	$(CC) -shared $(OBJS) -o $@

$(TEST_BIN): tests/test_runner.c $(STATIC_LIB) | build
	$(CC) $(CFLAGS) $< $(STATIC_LIB) -lm -o $@

test: all
	./$(TEST_BIN)

clean:
	rm -rf build
