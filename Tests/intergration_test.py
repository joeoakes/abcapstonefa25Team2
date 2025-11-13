import math
import pytest
import importlib.util
import sys
import os

# Dynamically import the module, regardless of its filename
module_path = os.path.abspath("01-shor_noiseless_demo.py")
spec = importlib.util.spec_from_file_location("shor_mod", module_path)
shor_mod = importlib.util.module_from_spec(spec)
sys.modules["shor_mod"] = shor_mod
spec.loader.exec_module(shor_mod)

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
        counts, factors = shor_mod.shor_factor_generic(a, N, shots=512)
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
    counts, factors = shor_mod.shor_factor_generic(a, N, shots=256)
    # Should not try quantum circuit, and factors should be None
    assert factors is None
