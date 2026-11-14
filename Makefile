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

# Where objects and binaries land. A variable, not a constant: an instrumented
# build must not overwrite the ordinary one. $(BUILD)/libbeam.a and the suites
# beside it are what verification/check.py reads, rebuilding them costs about
# an hour on a loaded box, and a ThreadSanitizer archive left at that path
# would be handed to the gate as though it were the shipping library. The
# `tsan` target sets this to build-tsan for that reason alone.
BUILD   ?= build

# Appended to both the compile and the link, and empty for an ordinary build.
# A sanitiser needs the same flag in both places. It is a separate variable
# rather than an override of CFLAGS so that an instrumented build ADDS to the
# shipping flags instead of restating them -- restating is how -mpopcnt or
# -Isrc/native/internal goes missing from one build, and that failure arrives
# as a wrong answer rather than as a compile error.
SANFLAGS ?=

# gcc spells it -fopenmp; clang against libomp needs -fopenmp=libomp.
OMPFLAG  ?= -fopenmp
CFLAGS   = -O2 -Wall -Wextra -std=c11 -mpopcnt -Iinclude -Isrc/native/internal
CFLAGS  += $(OMPFLAG) $(SANFLAGS)
LDLIBS   = -lm $(OMPFLAG) $(SANFLAGS)

# Strip comments and blank lines from the manifests.
read_manifest = $(shell sed -e 's/#.*//' -e '/^[[:space:]]*$$/d' $(1))

SRCS := $(call read_manifest,build_support/native_sources.txt)
OBJS := $(SRCS:%.c=$(BUILD)/%.o)
DEPS := $(OBJS:.o=.d)

# Fail loudly if the manifest names a file that is not there, rather than
# building a quietly incomplete archive.
MISSING := $(filter-out $(wildcard $(SRCS)),$(SRCS))
ifneq ($(MISSING),)
$(error source manifest lists missing file(s): $(MISSING))
endif

all: $(BUILD)/libbeam.a

$(BUILD)/libbeam.a: $(OBJS)
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
$(BUILD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# ── C unit tests ────────────────────────────────────────────────────────────
TEST_SRCS := $(wildcard tests/native/test_*.c)
TEST_BINS := $(TEST_SRCS:tests/native/%.c=$(BUILD)/tests/native/%)

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
$(BUILD)/tests/native/test_model: LDEXTRA = -Wl,--wrap=malloc

# The same mechanism used to MEASURE rather than to fail: this test sums what
# the allocator actually handed out during one analytic and checks the figure
# that analytic reports against it. Nothing else can catch an allocation that
# is simply never charged, because the account is otherwise the only witness.
$(BUILD)/tests/native/test_expected_counts_accounting: \
    LDEXTRA = -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc \
              -Wl,--wrap=free

# Every suite links the harness (tests/harness), which owns the counters, the
# outcomes, the per-test clock and the JSON. It is a source rather than an
# archive member so that a change to it rebuilds every suite through the same
# -MMD dependency tracking as everything else.
HARNESS_SRC := tests/harness/beam_test.c

# Every header a suite is compiled against, as a prerequisite of every suite.
# This file's opening paragraph says why that matters; the test binaries were
# the one place it did not hold. They are compiled and linked in ONE command,
# so there is no object file to hang -MMD on -- with several inputs gcc writes
# a dependency file per input, named after the input and dropped in the
# working directory. Listing the headers is coarser (any header rebuilds all
# 35) and exact, and there are three of them.
#
# The wildcard is deliberate: a header added to the harness or to the oracle
# layer is covered the day it arrives. The C ORACLE is why this changed --
# tests/harness/oracles/prior_reference.h had no witness at all, neither here
# nor in the gate's staleness check, which globbed tests/native/*.h.
TEST_HDRS := $(wildcard tests/native/*.h tests/harness/*.h \
                        tests/harness/oracles/*.h)

$(BUILD)/tests/native/%: tests/native/%.c $(HARNESS_SRC) $(TEST_HDRS) $(BUILD)/libbeam.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Itests/native -Itests/harness $< $(HARNESS_SRC) \
	    $(BUILD)/libbeam.a $(LDLIBS) $(LDEXTRA) -o $@

# The harness's own self-test. It fails on purpose, so it is deliberately not
# a dependency of `test`, and it lives outside tests/native so that neither
# TEST_SRCS nor the gate's inventory can see it.
$(BUILD)/tests/harness/%: tests/harness/%.c $(HARNESS_SRC) $(TEST_HDRS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Itests/harness $< $(HARNESS_SRC) $(LDLIBS) -o $@

selftest: $(BUILD)/tests/harness/selftest_outcomes


# ── ThreadSanitizer ─────────────────────────────────────────────────────────
#
# `make tsan` builds the library and the suites instrumented, into build-tsan/,
# leaving build/ untouched -- build/libbeam.a is what verification/check.py
# reads, and an instrumented archive left at that path would be handed to the
# gate as though it were the shipping library.
#
# IT MUST BE CLANG, AND libomp, AND ARCHER. This is not a preference, and the
# GCC path was tried first and measured, so that nobody has to try it again:
#
#   libgomp carries no sanitiser annotations (`nm -D libgomp.so.1` finds no
#   __tsan_* and no Annotate* symbol). TSan therefore sees NO happens-before
#   edge between any two OpenMP threads, so every implicit barrier is
#   invisible and every region that writes a result someone later reads is
#   reported as a race. Measured on test_distance alone, at 8 threads: 1186
#   reports, of which 1184 were main-thread-against-worker and 2 were
#   worker-against-worker -- and both of those two were ALSO barrier
#   artifacts, between pool threads reused across different regions
#   (advance._omp_fn.0 against advance._omp_fn.2). There is no filter that
#   separates a real intra-region race from a barrier-separated one when the
#   tool has no edges at all, which is why no such filter is shipped here.
#
# Archer is an OMPT tool that inserts exactly those edges. It is in Ubuntu's
# libomp-N-dev (/usr/lib/llvm-N/lib/libarcher.so), so this needs
# `apt install clang-N libomp-N-dev` and nothing built by hand.
#
# The optimisation level is deliberately NOT lowered. TSan instruments memory
# accesses, which survive -O2, and an instrumented build that tests different
# code from the one that ships answers a question nobody asked.
#
# Two suites are excluded, and must stay excluded: test_model and
# test_expected_counts_accounting link with -Wl,--wrap=malloc, while TSan
# replaces the allocator itself, so the wrapper and the sanitiser contend for
# one symbol. Both test allocation failure rather than parallelism, so nothing
# about threads is lost with them out.
LLVM_VER     ?= 19
LLVM_DIR     ?= /usr/lib/llvm-$(LLVM_VER)
TSAN_CC      ?= $(LLVM_DIR)/bin/clang
TSAN_ARCHER  ?= $(LLVM_DIR)/lib/libarcher.so
TSAN_EXCLUDE := test_model test_expected_counts_accounting
TSAN_SUITES  ?= $(filter-out $(TSAN_EXCLUDE),\
                    $(patsubst tests/native/%.c,%,$(TEST_SRCS)))

# Builds every suite without running one, which `test` and `list-tests` both
# do. The tsan target needs it, and so does anyone bisecting a compile error.
tests-only: $(TEST_BINS)

# Refuses rather than producing a log that cannot be read. A GCC-built
# instrumented binary runs perfectly well and reports a thousand races that
# are not there; failing here is the difference between no answer and a wrong
# one.
tsan:
	@test -x "$(TSAN_CC)" || { \
	    echo "make tsan: no clang at $(TSAN_CC)."; \
	    echo "  Install it:  sudo apt install clang-$(LLVM_VER) libomp-$(LLVM_VER)-dev"; \
	    echo "  Or point at another one:  make tsan LLVM_VER=21"; \
	    echo "  GCC is NOT a fallback -- see the comment above this target."; \
	    exit 1; }
	@test -f "$(TSAN_ARCHER)" || { \
	    echo "make tsan: clang is present but Archer is not ($(TSAN_ARCHER))."; \
	    echo "  Install it:  sudo apt install libomp-$(LLVM_VER)-dev"; \
	    echo "  Without Archer every barrier reads as a race."; \
	    exit 1; }
	@$(MAKE) --no-print-directory BUILD=build-tsan CC="$(TSAN_CC)" \
	    SANFLAGS="-fsanitize=thread -g -fno-omit-frame-pointer" \
	    OMPFLAG=-fopenmp=libomp \
	    $(addprefix build-tsan/tests/native/,$(TSAN_SUITES))

clean:
	rm -rf $(BUILD)

.PHONY: all clean test list-tests selftest tests-only tsan
-include $(DEPS)
