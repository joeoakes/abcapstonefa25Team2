import pytest
import importlib.util
import pathlib
import math
from qiskit import transpile
from qiskit_aer import Aer
# --- Dynamically import 01-shor_noiseless_demo.py ---
file_path = pathlib.Path(__file__).parent / "01-shor_noiseless_demo.py"
spec = importlib.util.spec_from_file_location("shor_module", file_path)
shor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shor_module)

# Pull functions from the module
build_shor_circuit = shor_module.build_shor_circuit
try_factors_from_r = shor_module.try_factors_from_r
# Some versions define this separately; handle if missing
try_order_from_measure = getattr(shor_module, "try_order_from_measure", None)


def test_shor_integration_qasm_simulator():
    """Run full Shor pipeline on Qiskit's QASM simulator and verify valid factorization."""
    a = 7
    N = 15
    t = 8
    shots = 256

    # --- Build and run circuit ---
    qc = build_shor_circuit(a=a, N=N, t=t)
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(qc, backend, optimization_level=1)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # --- Check the run actually produced results ---
    assert counts, "Simulator produced no measurement results."

    # --- Try to extract period r and compute factors ---
    success = False
    for bitstring, count in counts.items():
        meas_val = int(bitstring, 2)
        if try_order_from_measure:
            r = try_order_from_measure(meas_val, t, a, N)
            if r:
                factors = try_factors_from_r(a, r, N)
                if factors is not None:
                    p, q = factors
                    assert p * q == N, f"Expected factors to multiply to {N}"
                    success = True
                    break

    assert success, "No valid factors found; integration test failed."
