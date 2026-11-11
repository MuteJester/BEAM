"""The enumerated models both Mellin suites hold their transforms against.

`test_mellin_line.py` and `test_mellin_complex.py` each carried the same two
builders and the same fixture, verbatim: the same seeds, the same corpora, and
`log_pseq_batch` over every string in the space, twice. One fixture here, and
the builders in the shared engineering layer, where a second suite that needs
an enumerable model can find them.

Module-scoped and parametrised exactly as before, so every test id is
unchanged: one model per module per alphabet, built once.
"""
from __future__ import annotations

import pytest

from _engineering.fixtures import enumerated_aa_model, enumerated_nt_model


@pytest.fixture(scope="module", params=["nt", "aa"])
def enumerated(request):
    return (enumerated_nt_model() if request.param == "nt"
            else enumerated_aa_model())
