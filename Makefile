# BEAM -- dual-anchored expert model over CDR3 amino-acid sequences.
#
# Header dependency tracking (-MMD -MP) is not optional. Without it, changing
# a struct in a header leaves objects compiled against the old layout, and the
# result is a library that reads structures at the wrong offsets. That has
# been mistaken for undefined behaviour once and for a compiler defect once;
# both cost hours. See .private/ for the write-ups if they were carried over.

CC      ?= gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -Iinclude
CFLAGS  += -fopenmp
LDLIBS   = -lm -fopenmp

SRCS = $(wildcard lib/model/*.c) $(wildcard lib/core/*.c) \
       $(wildcard lib/occupancy/*.c)
OBJS = $(SRCS:%.c=build/%.o)
DEPS = $(OBJS:.o=.d)

all: build/libbeam.a

build/libbeam.a: $(OBJS)
	ar rcs $@ $^

build/%.o: %.c | build_dirs
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

build_dirs:
	@mkdir -p build/lib/model build/lib/core build/lib/occupancy \
	         build/tests/c_unit

# ── C unit tests ────────────────────────────────────────────────────────────
test: build/tests/c_unit/test_model build/tests/c_unit/test_publicness
	@./build/tests/c_unit/test_model
	@./build/tests/c_unit/test_publicness

# --wrap=malloc lets test_model make one allocation fail on demand, so
# the out-of-memory rollback paths are exercised rather than assumed. The
# wrapper forwards to the real allocator unless a test has armed it.
build/tests/c_unit/test_model: tests/c_unit/test_model.c build/libbeam.a | build_dirs
	$(CC) $(CFLAGS) -Ilib/model -Itests/c_unit $< build/libbeam.a $(LDLIBS) \
	    -Wl,--wrap=malloc -o $@

build/tests/c_unit/test_publicness: tests/c_unit/test_publicness.c build/libbeam.a | build_dirs
	$(CC) $(CFLAGS) -Itests/c_unit $< build/libbeam.a $(LDLIBS) -o $@

clean:
	rm -rf build

.PHONY: all clean test build_dirs
-include $(DEPS)
