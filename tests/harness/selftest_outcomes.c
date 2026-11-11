/**
 * @file selftest_outcomes.c
 * @brief The harness deciding outcomes, on cases built to be decided.
 *
 * NOT part of `make test`: it fails on purpose, and it lives outside
 * tests/native so that neither the suite (TEST_SRCS globs tests/native) nor
 * the gate's inventory (check_lib.c_inventory reads the same directory) can
 * see it. verification/tests/test_harness_outcomes.py builds and runs it.
 *
 * It exists because the harness now decides the outcome of all 390 C tests
 * from what their bodies did, and those precedence rules were the one part
 * of this QA work with nothing checking it.
 */
#include <stdio.h>

#include "beam_test.h"

static void selftest_a_pass_is_a_pass(void) {
    PASS();
}

static void selftest_an_assertion_failure_is_a_failure(void) {
    ASSERT_MSG(0, "deliberate: an assertion that cannot hold");
    PASS();                      /* unreachable: ASSERT_MSG returns */
}

static void selftest_a_skip_with_no_verdict_after_it_is_a_skip(void) {
    BEAM_SKIP("deliberate: nothing to verify here");
}

/* A sub-case that declines, exactly as test_row_cursor's does. */
static void a_sub_case_that_declines(void) {
    BEAM_SKIP("deliberate: one shape of several");
}

static void selftest_a_skip_followed_by_a_verdict_is_that_verdict(void) {
    a_sub_case_that_declines();
    PASS();
}

/*
 * Eleven test files increment fail_count directly in 74 places rather than
 * going through ASSERT_MSG. A harness that trusted only its own entry points
 * would record every one of those as having reached no verdict.
 */
static void selftest_a_direct_failure_count_is_a_failure(void) {
    fail_count++;
}

static void selftest_a_body_that_returns_silently_has_no_verdict(void) {
    return;
}

/*
 * Fifteen suites build their failure text with snprintf into ONE static
 * buffer and hand the harness a pointer to it. A harness that stores the
 * pointer rather than the text reports whatever was written there last --
 * for every record, since the JSON is written after the run. The printed
 * line is right and the machine-readable record, which is what the runner
 * reads, is wrong; a green suite can never show it, because it needs two
 * failures to appear.
 */
static char shared_buffer[96];

static void selftest_a_failure_message_is_the_text_that_was_there(void) {
    snprintf(shared_buffer, sizeof shared_buffer, "the first message");
    ASSERT_MSG(0, shared_buffer);
}

static void selftest_a_later_failure_does_not_rewrite_an_earlier_one(void) {
    snprintf(shared_buffer, sizeof shared_buffer, "the second message");
    ASSERT_MSG(0, shared_buffer);
}

/*
 * test_row_cursor is this shape: a helper declines one sub-case and the
 * caller carries on. If it then fails, the failure is what happened, and
 * the skip reason is not an explanation of it.
 */
static void selftest_a_failure_after_a_skip_says_why_it_failed(void) {
    beam_test_skip_here("one sub-case was declined");
    ASSERT_MSG(0, "the real failure");
}

int main(int argc, char **argv) {
    beam_test_init(argc, argv);
    RUN_TEST(selftest_a_pass_is_a_pass);
    RUN_TEST(selftest_an_assertion_failure_is_a_failure);
    RUN_TEST(selftest_a_skip_with_no_verdict_after_it_is_a_skip);
    RUN_TEST(selftest_a_skip_followed_by_a_verdict_is_that_verdict);
    RUN_TEST(selftest_a_direct_failure_count_is_a_failure);
    RUN_TEST(selftest_a_body_that_returns_silently_has_no_verdict);
    RUN_TEST(selftest_a_failure_message_is_the_text_that_was_there);
    RUN_TEST(selftest_a_later_failure_does_not_rewrite_an_earlier_one);
    RUN_TEST(selftest_a_failure_after_a_skip_says_why_it_failed);
    return beam_test_finish();
}
