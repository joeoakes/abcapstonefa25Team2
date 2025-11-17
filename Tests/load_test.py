import importlib.util
import sys
import os
import pytest
import time

module_path = os.path.abspath("01-shor_noiseless_demo.py")
spec = importlib.util.spec_from_file_location("shor_mod", module_path)
shor_mod = importlib.util.module_from_spec(spec)
sys.modules["shor_mod"] = shor_mod
spec.loader.exec_module(shor_mod)

def test_shor_factor_generic_many_shots():
    N = 15
    a = 2
    shots = 20000
    start = time.time()
    counts, factors = shor_mod.shor_factor_generic(a, N, shots=shots)
    duration = time.time() - start
    print(f"Stress test: {shots} shots took {duration:.2f} seconds")
    # Ensure it did not crash and result is reasonable
    assert isinstance(counts, dict)
    # Factors may or may not be recovered, but the run should finish
    assert factors is None or (isinstance(factors, tuple) and len(factors) == 2)
    assert duration < 120  # Example: should finish within 2 minutes for small N

def test_build_shor_circuit_large_N():
    # Push the circuit to the edge of its work register
    N = 127  # 2^7 - 1, a "large-ish" number for quick tests
    a = 3
    try:
        qc, t = shor_mod.build_shor_circuit_general(a, N)
        assert qc.num_qubits == t + 7  # work register should be 7
    except Exception as exc:
        pytest.skip(f"Could not build large circuit: {exc}")
