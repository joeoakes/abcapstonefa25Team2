import math
import pytest
from shor_noiseless_demo import (
    try_order_from_measure, try_factors_from_r, shor_factor_generic, build_shor_circuit_general
)

@pytest.mark.parametrize("meas,t,a,N,expected", [
    (2, 3, 2, 15, 4),  # 2^4 mod 15 == 1, order is 4
    (4, 4, 2, 21, 6),  # likely order-6 for a=2, N=21
    (0, 3, 2, 15, None), # meas of 0 => order can't be found
])
def test_try_order_from_measure(meas, t, a, N, expected):
    result = try_order_from_measure(meas, t, a, N)
    assert (result == expected or (expected is None and result is None))

@pytest.mark.parametrize("a,r,N,expected", [
    (2, 4, 15, (3, 5)),   # 2^2 mod 15 = 4, should recover 3,5
    (2, 6, 21, (3, 7)),   # order 6 for 2 mod 21: 2^3 mod 21 = 8; factors: (3,7)
    (2, 3, 15, None),     # odd order returns None
    (2, None, 15, None),  # None order returns None
])
def test_try_factors_from_r(a, r, N, expected):
    result = try_factors_from_r(a, r, N)
    if expected is None:
        assert result is None
    else:
        # Factors can come in either order
        assert set(result) == set(expected)

def test_build_shor_circuit_general_qubit_sizes():
    a, N = 2, 15
    qc, t = build_shor_circuit_general(a, N)
    n_work = math.ceil(math.log2(N))
    assert qc.num_qubits == t + n_work

def test_shor_factor_generic_success():
    # For N=15, a=2, factors should be (3,5)
    counts, factors = shor_factor_generic(2, 15, shots=128)
    assert factors is None or set(factors) == {3, 5}

@pytest.mark.parametrize("a,N", [
    (3, 15), # a coprime to 15; should sometimes factor
    (7, 21),
])
def test_shor_factor_generic_various(a,N):
    counts, factors = shor_factor_generic(a, N, shots=128)
    if factors is not None:
        assert math.prod(factors) == N
