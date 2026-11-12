/**
 * @file test_avail.c
 * @brief Tests for the free-core estimate the thread policy is given.
 *
 * The parsers take their text as an ARGUMENT rather than reading /proc and
 * /sys themselves, which is the only way to test them: a test cannot create a
 * cgroup, and a test that reads the real machine asserts whatever the machine
 * happens to be doing. Everything deterministic is therefore pure, and only
 * the thin composition that reads the two files is not.
 *
 * The expected values are written by hand from real captured text, including
 * the captures from a saturated 48-core cluster node, so a transposed field
 * index shows up as a wrong number rather than as agreement with the bug.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "beam/common.h"
#include "beam/avail.h"

#include "test_utils.h"

/* ── The runnable count, from /proc/loadavg's fourth field ───── */

static void test_the_runnable_count_is_read_from_the_fourth_field(void) {
    /*
     * Captured text, not synthesised. The first is this development box when
     * idle; the second is ig05 while 99.9% busy on 48 cores, where the
     * runnable count of 64 is what correctly says "oversubscribed" and the
     * 1-minute average of 71.26 only says so by accident.
     */
    ASSERT_MSG(beam_parse_nr_running("0.56 0.75 6.85 2/3361 882022") == 2,
               "an idle box reads 2 runnable");
    ASSERT_MSG(beam_parse_nr_running("71.26 68.37 63.32 64/1544 1311698") == 64,
               "a saturated 48-core node reads 64 runnable");
    ASSERT_MSG(beam_parse_nr_running("3.70 1.20 0.40 16/900 1\n") == 16,
               "a trailing newline must not change the answer");
    PASS();
}

static void test_an_unreadable_loadavg_is_reported_as_unknown(void) {
    /*
     * -1 means "could not tell", which the arithmetic below turns into "assume
     * no contention" -- and the policy then applies its headroom rule alone.
     * That is deliberately the same answer a machine with no /proc gets.
     */
    const char *bad[] = {
        "", "garbage", "0.5 0.6", "0.5 0.6 0.7",
        "0.5 0.6 0.7 xx/3361", "0.5 0.6 0.7 /3361", "0.5 0.6 0.7 -4/3361",
    };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
        const int got = beam_parse_nr_running(bad[i]);
        if (got != -1) {
            char msg[160];
            snprintf(msg, sizeof msg,
                     "%s should be unknown (-1), got %d", bad[i], got);
            beam_test_fail(msg);
            return;
        }
    }
    ASSERT_MSG(beam_parse_nr_running(NULL) == -1, "NULL text is unknown");
    PASS();
}

/* ── The cgroup v2 CPU quota, from cpu.max ───────────────────── */

static void test_the_cgroup_quota_is_read_as_whole_cores(void) {
    ASSERT_MSG(beam_parse_cpu_max("200000 100000") == 2,
               "two periods' worth of quota is two cores");
    ASSERT_MSG(beam_parse_cpu_max("800000 100000\n") == 8,
               "eight cores, with a trailing newline");
    /*
     * A FRACTIONAL quota floors rather than rounds up. 1.5 cores of quota
     * cannot run two threads without being throttled, and the asymmetry
     * measured in phase 0 is severe -- planning too wide cost up to 3814x,
     * planning too narrow costs about 2x -- so the conservative direction is
     * the correct one here.
     */
    ASSERT_MSG(beam_parse_cpu_max("150000 100000") == 1,
               "1.5 cores of quota floors to one");
    ASSERT_MSG(beam_parse_cpu_max("50000 100000") == 1,
               "half a core of quota is still at least one thread");
    PASS();
}

static void test_no_cgroup_limit_is_reported_as_unknown(void) {
    /* "max" is the literal cgroup v2 spelling for "no limit". */
    ASSERT_MSG(beam_parse_cpu_max("max 100000") == -1,
               "an unlimited cgroup must not cap anything");
    ASSERT_MSG(beam_parse_cpu_max("max") == -1, "bare max is also unlimited");
    PASS();
}

static void test_an_unreadable_cpu_max_is_reported_as_unknown(void) {
    /*
     * This is review-focus item 5, as a test. A misparse that returned a
     * large number would be acted on; one that returned 0 would serialise
     * everything. Both are worse than admitting ignorance.
     */
    const char *bad[] = { "", "abc", "100000", "100000 0", "abc 100000",
                          "100000 abc", "-100000 100000" };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
        const int got = beam_parse_cpu_max(bad[i]);
        if (got != -1) {
            char msg[160];
            snprintf(msg, sizeof msg,
                     "%s should be unknown (-1), got %d", bad[i], got);
            beam_test_fail(msg);
            return;
        }
    }
    ASSERT_MSG(beam_parse_cpu_max(NULL) == -1, "NULL text is unknown");
    PASS();
}

/* ── Composing the estimate ──────────────────────────────────── */

static void test_free_cores_is_what_is_allowed_minus_what_is_running(void) {
    /*
     * One is subtracted for the caller itself, which is runnable by definition
     * at the moment it reads the file. The measured bias is about +2 rather
     * than +1 -- there is usually one more runnable entity on a desktop -- and
     * that remainder is left in, because it is real contention.
     */
    ASSERT_MSG(beam_available_from(16, 1) == 16,
               "only ourselves runnable means the machine is ours");
    ASSERT_MSG(beam_available_from(16, 11) == 6,
               "ten competitors on sixteen cores leaves six");
    ASSERT_MSG(beam_available_from(48, 64) == 1,
               "a saturated node leaves one, never zero or negative");
    PASS();
}

static void test_unknown_contention_falls_back_to_the_headroom_rule(void) {
    /*
     * With no runnable count -- no /proc, as on Windows -- the estimate is
     * everything allowed, and the policy's own halving then yields exactly the
     * conservative constant that was wanted for that case. The fallback is not
     * a special case anywhere; it falls out of the existing rule.
     */
    ASSERT_MSG(beam_available_from(16, -1) == 16,
               "unknown contention assumes none and lets the policy halve it");
    ASSERT_MSG(beam_available_from(16, 0) == 16,
               "a nonsensical zero runnable count is treated as unknown");
    PASS();
}

static void test_free_cores_is_never_below_one_or_above_what_is_allowed(void) {
    for (int allowed = -4; allowed <= 64; ++allowed) {
        for (int nr = -4; nr <= 80; ++nr) {
            const int got = beam_available_from(allowed, nr);
            const int cap = allowed < 1 ? 1 : allowed;
            if (got < 1 || got > cap) {
                char msg[160];
                snprintf(msg, sizeof msg,
                         "allowed=%d nr=%d gave %d, outside [1, %d]",
                         allowed, nr, got, cap);
                beam_test_fail(msg);
                return;
            }
        }
    }
    PASS();
}

static void test_more_competitors_never_frees_more_cores(void) {
    /* Monotone the other way from the policy's rule, and for the same reason:
     * the answer has to be explicable to someone watching their machine. */
    int prev = 1 << 30;
    for (int nr = 1; nr <= 200; ++nr) {
        const int got = beam_available_from(32, nr);
        if (got > prev) {
            char msg[160];
            snprintf(msg, sizeof msg,
                     "nr=%d freed %d cores, more than %d at nr=%d",
                     nr, got, prev, nr - 1);
            beam_test_fail(msg);
            return;
        }
        prev = got;
    }
    PASS();
}

/* ── Reading the real machine ────────────────────────────────── */

static void test_the_machine_reports_a_usable_estimate(void) {
    /*
     * The one test that touches the real system, so it asserts only what must
     * hold on ANY machine: a positive answer, no larger than the cores this
     * process is allowed to run on. Asserting a particular number here would
     * be asserting what the box happened to be doing.
     */
    const int allowed = beam_allowed_cores();
    const int avail = beam_available_cores();
    char msg[160];
    snprintf(msg, sizeof msg, "allowed=%d must be at least 1", allowed);
    ASSERT_MSG(allowed >= 1, msg);
    snprintf(msg, sizeof msg, "available=%d must be in [1, %d]", avail, allowed);
    ASSERT_MSG(avail >= 1 && avail <= allowed, msg);
    PASS();
}

static void test_the_estimate_does_not_exceed_the_cores_present(void) {
    /*
     * An affinity mask or a cgroup quota can only ever narrow what the machine
     * has. A larger answer would mean a misparse being acted on, which is the
     * failure this file's unknown-is-minus-one convention exists to prevent.
     */
    const int present = beam_cores_present();
    const int allowed = beam_allowed_cores();
    char msg[160];
    snprintf(msg, sizeof msg,
             "allowed=%d must not exceed the %d cores present", allowed, present);
    ASSERT_MSG(present >= 1, "the machine must report at least one core");
    ASSERT_MSG(allowed <= present, msg);
    PASS();
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    beam_test_init(argc, argv);
    printf("\nBEAM Unit Tests - Free-core estimate\n");
    printf("==============================================================\n\n");

    RUN_TEST(test_the_runnable_count_is_read_from_the_fourth_field);
    RUN_TEST(test_an_unreadable_loadavg_is_reported_as_unknown);
    RUN_TEST(test_the_cgroup_quota_is_read_as_whole_cores);
    RUN_TEST(test_no_cgroup_limit_is_reported_as_unknown);
    RUN_TEST(test_an_unreadable_cpu_max_is_reported_as_unknown);
    RUN_TEST(test_free_cores_is_what_is_allowed_minus_what_is_running);
    RUN_TEST(test_unknown_contention_falls_back_to_the_headroom_rule);
    RUN_TEST(test_free_cores_is_never_below_one_or_above_what_is_allowed);
    RUN_TEST(test_more_competitors_never_frees_more_cores);
    RUN_TEST(test_the_machine_reports_a_usable_estimate);
    RUN_TEST(test_the_estimate_does_not_exceed_the_cores_present);

    printf("\n==============================================================\n");
    return beam_test_finish();
}
