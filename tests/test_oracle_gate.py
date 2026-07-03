"""The gate. Once the harness is implemented, this asserts the oracle covers ~95%.

Until then it is skipped. This test is the guardrail: neural generators do not
get built until the oracle passes here.
"""
import pytest


@pytest.mark.skip(reason="Implement after the harness and combining rules exist.")
def test_oracle_covers_nominal():
    # Run the oracle config for E1-E3; assert coverage within, say, [0.93, 0.97].
    ...
