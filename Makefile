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
CFLAGS   = -O2 -Wall -Wextra -std=c11 -Iinclude -Isrc/native/internal
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

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t || exit 1; done

# --wrap=malloc lets test_model make one allocation fail on demand, so the
# out-of-memory rollback paths are exercised rather than assumed. It is
# target-specific and must stay that way: --wrap redirects EVERY malloc in the
# linked objects to __wrap_malloc, so applying it to a binary that does not
# define that symbol fails to link.
build/tests/native/test_model: LDEXTRA = -Wl,--wrap=malloc

build/tests/native/%: tests/native/%.c build/libbeam.a
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -Itests/native $< build/libbeam.a $(LDLIBS) \
	    $(LDEXTRA) -o $@

clean:
	rm -rf build

.PHONY: all clean test
-include $(DEPS)
