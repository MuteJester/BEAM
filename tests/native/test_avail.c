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
#define _GNU_SOURCE          /* setenv/unsetenv under -std=c11 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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
     * With no runnable count the estimate is everything allowed, and clause 3
     * of the policy then caps it at the scaling ceiling -- so an unknown
     * machine is planned at 8. The fallback is not a special case anywhere; it
     * falls out of the existing rule.
     *
     * This comment used to credit "the policy's own halving" with that answer,
     * which plan.c stopped doing when the ceiling form replaced the halving.
     * It also used to say "as on Windows", which was the whole gap: Windows
     * and macOS answered unknown on EVERY call, so they were planned at 8
     * whatever the machine was doing. They now difference two CPU-time
     * readings instead, and reach this path only until a window accumulates.
     */
    ASSERT_MSG(beam_available_from(16, -1) == 16,
               "unknown contention assumes none and lets the policy halve it");
    ASSERT_MSG(beam_available_from(16, 0) == 16,
               "a nonsensical zero runnable count is treated as unknown");
    PASS();
}

static void test_a_window_too_short_to_mean_anything_is_unknown(void) {
    /*
     * CPU time is accounted in ticks -- about 10 ms on both platforms that
     * need this -- so a window of a few microseconds between two BEAM calls
     * carries no information at all: the deltas are zero or one tick and the
     * quotient is noise. Unknown, which the policy already handles as "assume
     * no contention", is the honest answer and the existing safe fallback.
     */
    ASSERT_MSG(beam_competitors_from_cpu_time(0u, 0u, 1000u, 16) == -1,
               "a one-microsecond window cannot be measured");
    ASSERT_MSG(beam_competitors_from_cpu_time(100000000u, 0u, 5000000u, 16) == -1,
               "five milliseconds is still below one accounting tick");
    PASS();
}

static void test_an_idle_machine_reports_only_ourselves(void) {
    /* One second of wall, one core's worth of CPU, all of it ours. */
    ASSERT_MSG(beam_competitors_from_cpu_time(1000000000u, 1000000000u,
                                              1000000000u, 16) == 1,
               "our own core does not compete with us");
    PASS();
}

static void test_competitors_are_counted_in_whole_cores(void) {
    /*
     * One second of wall during which the machine burned five core-seconds,
     * one of them ours: four competitors, plus ourselves, is five runnable --
     * the same units beam_available_from takes.
     */
    ASSERT_MSG(beam_competitors_from_cpu_time(5000000000u, 1000000000u,
                                              1000000000u, 16) == 5,
               "four competing cores plus ourselves");
    ASSERT_MSG(beam_available_from(16, beam_competitors_from_cpu_time(
                   5000000000u, 1000000000u, 1000000000u, 16)) == 12,
               "and that leaves twelve of sixteen");
    PASS();
}

static void test_our_own_threads_are_not_competition(void) {
    /*
     * THE reason this function subtracts. A tick difference spanning BEAM's
     * own eight-thread region reads the machine as eight cores busy, and
     * without the subtraction the next call would plan narrow BECAUSE OF THE
     * PREVIOUS ONE -- the feedback loop avail.h warns about for the load
     * average, which "would serialise every call for a minute after BEAM's own
     * previous call finished". Eight core-seconds, all ours, is an idle
     * machine as far as the plan is concerned.
     */
    ASSERT_MSG(beam_competitors_from_cpu_time(8000000000u, 8000000000u,
                                              1000000000u, 16) == 1,
               "our own eight threads leave the machine ours");
    PASS();
}

static void test_a_saturated_machine_cannot_report_more_than_it_has(void) {
    /*
     * System CPU time covers every CPU, while `allowed` may be a subset --
     * the same approximation the runnable count already makes on Linux. The
     * clamp keeps a hundred busy cores from producing a negative estimate,
     * and the policy's own floor then yields one.
     */
    const int nr = beam_competitors_from_cpu_time(100000000000u, 0u,
                                                  1000000000u, 16);
    ASSERT_MSG(nr == 17, "sixteen allowed, so at most sixteen competitors plus us");
    ASSERT_MSG(beam_available_from(16, nr) == 1, "a saturated machine leaves one");
    PASS();
}

static void test_a_fraction_of_a_core_rounds_towards_contention(void) {
    /*
     * Half a core of competition rounds up to one, not down to none. Planning
     * too narrow costs about 2x and planning too wide cost up to 3814x, so
     * the rounding goes the cheap way.
     */
    ASSERT_MSG(beam_competitors_from_cpu_time(1500000000u, 0u,
                                              1000000000u, 16) == 3,
               "one and a half competing cores counts as two, plus ourselves");
    PASS();
}

static void test_an_inconsistent_sample_is_not_negative_competition(void) {
    /*
     * The two clocks are read one after the other, not atomically, so our own
     * CPU time can exceed the system total the sample before it. Reporting
     * fewer than zero competitors would widen the plan on the strength of a
     * sampling race.
     */
    ASSERT_MSG(beam_competitors_from_cpu_time(1000000000u, 4000000000u,
                                              1000000000u, 16) == 1,
               "more of our own time than the system's means no competition");
    PASS();
}

static void test_a_threads_first_reading_is_only_a_baseline(void) {
    /* Nothing to difference against, so unknown -- and the reading is kept. */
    BeamAvailSample s = {0};
    ASSERT_MSG(beam_competitors_from_samples(&s, 5000000000u, 1000000000u,
                                             1000000000u, 16) == -1,
               "the first reading cannot be interpreted");
    ASSERT_MSG(s.have == 1, "but it becomes the baseline");
    ASSERT_MSG(s.wall == 1000000000u, "stored as given");
    PASS();
}

static void test_a_short_window_leaves_the_baseline_to_age(void) {
    /*
     * The rule the whole mechanism depends on. A tight loop of short calls
     * must not restart the window each time, or twenty milliseconds is never
     * reached and the estimate stays unknown on exactly the workload that
     * needs it. Three five-millisecond calls report unknown and leave the
     * baseline where it was; the fourth, now twenty-five milliseconds from
     * that baseline, answers.
     */
    BeamAvailSample s = {0};
    (void)beam_competitors_from_samples(&s, 0u, 0u, 1000000000u, 16);
    const uint64_t base = s.wall;
    for (int i = 1; i <= 3; i++) {
        const uint64_t at = 1000000000u + (uint64_t)i * 5000000u;
        ASSERT_MSG(beam_competitors_from_samples(&s, 0u, 0u, at, 16) == -1,
                   "five milliseconds cannot be measured");
        ASSERT_MSG(s.wall == base, "and the baseline must not move");
    }
    const int nr = beam_competitors_from_samples(&s, 50000000u, 0u,
                                                 1025000000u, 16);
    ASSERT_MSG(nr == 3, "two competing cores plus ourselves, once the window is real");
    ASSERT_MSG(s.wall == 1025000000u, "and now the baseline advances");
    PASS();
}

static void test_a_reading_that_moves_backwards_resets_rather_than_underflows(void) {
    /*
     * The clocks promise monotonicity; this pins what happens if one breaks
     * it, because the subtraction is unsigned and the failure would not be a
     * wrong answer but a vast one.
     */
    BeamAvailSample s = {0};
    (void)beam_competitors_from_samples(&s, 9000000000u, 2000000000u,
                                        5000000000u, 16);
    ASSERT_MSG(beam_competitors_from_samples(&s, 9000000000u, 2000000000u,
                                             4000000000u, 16) == -1,
               "a wall clock that went backwards is unknown, not enormous");
    ASSERT_MSG(beam_competitors_from_samples(&s, 1000000000u, 2000000000u,
                                             9000000000u, 16) == -1,
               "system time that went backwards is unknown too");
    PASS();
}

static void test_the_sampled_estimate_tracks_competitors_over_the_window(void) {
    /*
     * End to end through the stateful step: one second of wall in which the
     * machine burned twelve core-seconds, four of them BEAM's own, leaves
     * eight competitors and so eight of sixteen cores free.
     */
    BeamAvailSample s = {0};
    (void)beam_competitors_from_samples(&s, 0u, 0u, 0u, 16);
    const int nr = beam_competitors_from_samples(&s, 12000000000u, 4000000000u,
                                                 1000000000u, 16);
    ASSERT_MSG(nr == 9, "eight competing cores plus ourselves");
    ASSERT_MSG(beam_available_from(16, nr) == 8, "which leaves eight free");
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

/* ── Pinning the estimate ────────────────────────────────────── */

static void test_a_pinned_estimate_overrides_the_measurement(void) {
    /*
     * A load-aware plan is not reproducible between runs, which the design
     * accepted as the cost of being right about a busy machine. The pin is how
     * a benchmark, a CI run or a test gets determinism back: it is the same
     * role set_threads() plays for the cap.
     */
    setenv("BEAM_AVAILABLE_CORES", "4", 1);
    const int pinned = beam_available_cores();
    unsetenv("BEAM_AVAILABLE_CORES");
    ASSERT_MSG(pinned == 4, "a pin of 4 must be reported as 4 free cores");
    PASS();
}

static void test_a_nonsense_pin_is_ignored_rather_than_obeyed(void) {
    /*
     * A typo in an environment variable must not be able to serialise a
     * library or to claim a thousand free cores. Each of these falls back to
     * measuring, which always yields at least one.
     */
    const char *bad[] = { "", "abc", "0", "-5", "4x", " " };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; ++i) {
        setenv("BEAM_AVAILABLE_CORES", bad[i], 1);
        const int got = beam_available_cores();
        const int allowed = beam_allowed_cores();
        unsetenv("BEAM_AVAILABLE_CORES");
        if (got < 1 || got > allowed) {
            char msg[160];
            snprintf(msg, sizeof msg,
                     "pin %s gave %d, outside the measured range [1, %d]",
                     bad[i], got, allowed);
            beam_test_fail(msg);
            return;
        }
    }
    PASS();
}

static void test_a_pin_above_the_cores_present_is_clamped(void) {
    setenv("BEAM_AVAILABLE_CORES", "100000", 1);
    const int got = beam_available_cores();
    const int present = beam_cores_present();
    unsetenv("BEAM_AVAILABLE_CORES");
    char msg[160];
    snprintf(msg, sizeof msg, "a pin of 100000 gave %d, above the %d present",
             got, present);
    ASSERT_MSG(got <= present, msg);
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
    RUN_TEST(test_a_window_too_short_to_mean_anything_is_unknown);
    RUN_TEST(test_an_idle_machine_reports_only_ourselves);
    RUN_TEST(test_competitors_are_counted_in_whole_cores);
    RUN_TEST(test_our_own_threads_are_not_competition);
    RUN_TEST(test_a_saturated_machine_cannot_report_more_than_it_has);
    RUN_TEST(test_a_fraction_of_a_core_rounds_towards_contention);
    RUN_TEST(test_an_inconsistent_sample_is_not_negative_competition);
    RUN_TEST(test_a_threads_first_reading_is_only_a_baseline);
    RUN_TEST(test_a_short_window_leaves_the_baseline_to_age);
    RUN_TEST(test_a_reading_that_moves_backwards_resets_rather_than_underflows);
    RUN_TEST(test_the_sampled_estimate_tracks_competitors_over_the_window);
    RUN_TEST(test_free_cores_is_never_below_one_or_above_what_is_allowed);
    RUN_TEST(test_more_competitors_never_frees_more_cores);
    RUN_TEST(test_the_machine_reports_a_usable_estimate);
    RUN_TEST(test_the_estimate_does_not_exceed_the_cores_present);
    RUN_TEST(test_a_pinned_estimate_overrides_the_measurement);
    RUN_TEST(test_a_nonsense_pin_is_ignored_rather_than_obeyed);
    RUN_TEST(test_a_pin_above_the_cores_present_is_clamped);

    printf("\n==============================================================\n");
    return beam_test_finish();
}
