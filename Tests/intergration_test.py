import math
import pytest
from shor_noiseless_demo import shor_factor_generic

@pytest.mark.parametrize(
    "N,a,expected_factors",
    [
        (15, 2, {3, 5}),
        (21, 2, {3, 7}),
        (21, 5, {3, 7}),
        (35, 3, {5, 7}),
    ]
)
def test_shor_factor_integration(N, a, expected_factors):
    # Try multiple times due to probabilistic outcome
    found = False
    for _ in range(5):
        counts, factors = shor_factor_generic(a, N, shots=512)
        if factors is not None and set(factors) == expected_factors:
            found = True
            break
    assert found, f"Failed to find factors {expected_factors} of N={N} with a={a}, got: {factors}"

@pytest.mark.parametrize(
    "N,a",
    [
        (15, 5),   # a shares a factor with N
        (21, 7),   # a shares a factor with N
        (35, 5),
        (143, 11),
    ]
)
def test_shor_factor_integration_non_coprime(N, a):
    counts, factors = shor_factor_generic(a, N, shots=256)
    # Should not try quantum circuit, and factors should be None
    assert factors is None
