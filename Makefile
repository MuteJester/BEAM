# BEAM -- Bidirectional Expert-Anchored Model.
#
# Header dependency tracking (-MMD -MP) is not optional. Without it, changing
# a struct in a header leaves objects compiled against the old layout, and the
# result is a library that reads structures at the wrong offsets. That has
# been mistaken for undefined behaviour once and for a compiler defect once;
# both cost hours.
#
# Sources come from explicit manifests under build_support/, not from
# wildcards. See build_support/native_sources.txt for why.

CC      ?= gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -mpopcnt -Iinclude -Isrc/native/internal
CFLAGS  += -fopenmp
LDLIBS   = -lm -fopenmp

# Strip comments and blank lines from the manifests.
read_manifest = $(shell sed -e 's/#.*//' -e '/^[[:space:]]*$$/d' $(1))

SRCS := $(call read_manifest,build_support/native_sources.txt)
OBJS := $(SRCS:%.c=build/%.o)
DEPS := $(OBJS:.o=.d)

# Fail loudly if the manifest names a file that is not there, rather than
# building a quietly incomplete archive.
MISSING := $(filter-out $(wildcard $(SRCS)),$(SRCS))
ifneq ($(MISSING),)
$(error source manifest lists missing file(s): $(MISSING))
endif

all: build/libbeam.a

build/libbeam.a: $(OBJS)
	@mkdir -p $(@D)
# Deleted before it is created, because `ar rcs` ADDS to an existing archive
# rather than replacing it. A source removed from the manifest leaves its
# object behind in the .a indefinitely, exporting symbols no source defines --
# which is invisible until something reads the archive's symbol table, as
# verification/check.py does. One such ghost was found in a working tree that
# had not been `make clean`ed, and would have been recorded into the baseline
# as public API.
	@rm -f $@
	ar rcs $@ $^

# The target directory is created per object, so nested source layouts need no
# hard-coded directory list here.
build/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# ── C unit tests ────────────────────────────────────────────────────────────
TEST_SRCS := $(wildcard tests/native/test_*.c)
TEST_BINS := $(TEST_SRCS:tests/native/%.c=build/tests/native/%)

# tools/qa/run reports every suite and continues past a failure. The loop
# this replaces was `for t in $(TEST_BINS); do ./$$t || exit 1; done`, which
# stopped at the first failing suite and hid up to 34 others, and let every
# binary claim all 16 cores -- at any real load that is thrashing, not
# parallelism. The runner caps threads per suite and runs several suites at
# once instead.
test: $(TEST_BINS)
	@tools/qa/run c

# verification/check.py reads the C test inventory by asking each binary what
# it runs, rather than by grepping RUN_TEST out of these sources -- see
# check_lib.c_inventory for why. This target asks the same question by hand,
# so the gate's input is reproducible without reading the gate, and so a
# named suite's contents can be seen without running it.
list-tests: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "== $$t"; ./$$t --list; done

# --wrap=malloc lets test_model make one allocation fail on demand, so the
# out-of-memory rollback paths are exercised rather than assumed. It is
# target-specific and must stay that way: --wrap redirects EVERY malloc in the
# linked objects to __wrap_malloc, so applying it to a binary that does not
# define that symbol fails to link.
build/tests/native/test_model: LDEXTRA = -Wl,--wrap=malloc

# The same mechanism used to MEASURE rather than to fail: this test sums what
# the allocator actually handed out during one analytic and checks the figure
# that analytic reports against it. Nothing else can catch an allocation that
# is simply never charged, because the account is otherwise the only witness.
build/tests/native/test_expected_counts_accounting: \
    LDEXTRA = -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc \
              -Wl,--wrap=free

# Every suite links the harness (tests/harness), which owns the counters, the
# outcomes, the per-test clock and the JSON. It is a source rather than an
# archive member so that a change to it rebuilds every suite through the same
# -MMD dependency tracking as everything else.
HARNESS_SRC := tests/harness/beam_test.c
HARNESS_HDR := tests/harness/beam_test.h

build/tests/native/%: tests/native/%.c $(HARNESS_SRC) $(HARNESS_HDR) build/libbeam.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Itests/native -Itests/harness $< $(HARNESS_SRC) \
	    build/libbeam.a $(LDLIBS) $(LDEXTRA) -o $@

# The harness's own self-test. It fails on purpose, so it is deliberately not
# a dependency of `test`, and it lives outside tests/native so that neither
# TEST_SRCS nor the gate's inventory can see it.
build/tests/harness/%: tests/harness/%.c $(HARNESS_SRC) $(HARNESS_HDR)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Itests/harness $< $(HARNESS_SRC) $(LDLIBS) -o $@

selftest: build/tests/harness/selftest_outcomes

clean:
	rm -rf build

.PHONY: all clean test list-tests selftest
-include $(DEPS)
