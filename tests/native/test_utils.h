/**
 * @file test_utils.h
 * @brief What the test files include: the harness, under the old name.
 *
 * The harness itself is tests/harness/beam_test.{h,c}. This file stays
 * because 35 test files include it by name. It used to carry the
 * hierarchical prior restated from its definition as well; that is an oracle,
 * not a harness, and it now lives in tests/harness/oracles/prior_reference.h,
 * included directly by the three suites that check a finalized cell against
 * the counts it came from.
 */
#ifndef BEAM_TEST_UTILS_H
#define BEAM_TEST_UTILS_H

#include "beam_test.h"


#endif /* BEAM_TEST_UTILS_H */
