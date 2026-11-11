/**
 * @file beam_test.h
 * @brief BEAM's C test harness: the vocabulary.
 *
 * The counters live in the harness, not in the test files. Thirty-four files
 * each declared their own `static int pass_count = 0, fail_count = 0;`, which
 * is how two files could disagree about what counts as a test and how a
 * summary could be printed in three different formats.
 *
 * A test has one of four outcomes, and the harness decides which from what
 * the test actually did:
 *
 *   failed      something recorded a failure
 *   passed      PASS() was reached and nothing failed
 *   skipped     BEAM_SKIP said why, and no verdict followed
 *   no verdict  the body returned having neither passed nor failed
 *
 * In that order of precedence. A test that skips one sub-case and then
 * verifies the others has PASSED -- reporting it skipped under-reports it
 * exactly as counting a skip as a pass over-reports it. "No verdict" is a
 * defect in the test and fails the run: before this harness it was invisible,
 * because a body that returned early simply made the pass count smaller.
 *
 * Command line, through beam_test_init(argc, argv) as main's first statement:
 *
 *   --list         print the inventory, one name per line, and run nothing
 *   --run NAME     run only NAME. Repeatable. A name this binary does not
 *                  have is an error, not an empty success
 *   --json PATH    write the outcomes, their durations and the conditions
 */
#ifndef BEAM_TEST_H
#define BEAM_TEST_H

#include <stdio.h>

/* The counters, owned here. The names are the ones the test files use. */
extern int pass_count;
extern int fail_count;
extern int beam_test_list_only;

void beam_test_init(int argc, char **argv);
int  beam_test_finish(void);
void beam_test_report_json(const char *path);

/* Used by the macros below; call them through the macros. */
int  beam_test_wanted(const char *name);
void beam_test_begin(const char *name);
void beam_test_end(void);
void beam_test_pass(void);
void beam_test_fail(const char *message);
void beam_test_skip_here(const char *reason);
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
int  beam_test_printf(const char *fmt, ...);

/*
 * While listing, stdout carries the inventory and nothing else: it is the
 * input to verification/check.py and to `make list-tests`, and anything that
 * has to pick names out of banner text can silently drop one. The test files
 * print their own section banners with printf, so printf is routed through a
 * wrapper that goes quiet while listing. Names are emitted with fputs.
 */
#define printf(...) beam_test_printf(__VA_ARGS__)

#define RUN_TEST(fn) do {                                        \
    if (beam_test_list_only) { fputs(#fn "\n", stdout); break; } \
    if (!beam_test_wanted(#fn)) break;                           \
    beam_test_begin(#fn);                                        \
    fn();                                                        \
    beam_test_end();                                             \
} while (0)

#define ASSERT_MSG(cond, msg) do { \
    if (!(cond)) { beam_test_fail(msg); return; } \
} while (0)

#define PASS() beam_test_pass()

/* Say why, and stop. Whether the TEST is skipped depends on whether a
 * verdict follows: see the precedence above. */
#define BEAM_SKIP(reason) do { beam_test_skip_here(reason); return; } while (0)

#endif /* BEAM_TEST_H */
