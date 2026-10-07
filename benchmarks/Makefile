# The native benchmark harness. Run from the repository root:
#
#     make -f benchmarks/Makefile
#
# Same compiler flags as the library and the Python extension (-O2 -mpopcnt
# -fopenmp), plus -g so `perf record` can name what it samples; -g does not
# change the generated code.

CC      ?= gcc
CFLAGS   = -O2 -g -Wall -Wextra -std=c11 -D_GNU_SOURCE -mpopcnt -fopenmp \
           -Iinclude -Isrc/native/internal -Ibenchmarks/c
LDLIBS   = -lm -fopenmp
# The heap tracker in bench_memory.c sees every allocation libbeam.a makes.
WRAP     = -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=aligned_alloc \
           -Wl,--wrap=free

# Suites are discovered: a new file in benchmarks/c/suites/ is built and its
# BEAM_BENCHMARKs are registered with no other edit.
BENCH_SRCS := benchmarks/c/main.c benchmarks/c/bench_memory.c benchmarks/c/bench_fixtures.c \
              $(sort $(wildcard benchmarks/c/suites/*.c))
BENCH_OBJS := $(BENCH_SRCS:%.c=build/%.o)
TARGET     := build/benchmarks/beam_bench

all: $(TARGET)

# Always ask the root Makefile, which tracks every library source and header.
# A prerequisite-free rule here would accept any existing archive as current,
# and the harness would go on measuring the library as it was before an edit.
# Make re-reads the archive's timestamp afterwards, so the harness is relinked
# only when the archive actually changed.
build/libbeam.a: FORCE
	@$(MAKE) --no-print-directory -f Makefile build/libbeam.a

# -MMD -MP: a changed header recompiles every benchmark object that includes
# it, including BEAM's public headers, whose struct layouts the suites use.
build/benchmarks/%.o: benchmarks/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(BENCH_OBJS) build/libbeam.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(BENCH_OBJS) build/libbeam.a $(WRAP) $(LDLIBS) -o $@

clean:
	rm -rf build/benchmarks

-include $(BENCH_OBJS:.o=.d)

.PHONY: all clean FORCE
FORCE:
