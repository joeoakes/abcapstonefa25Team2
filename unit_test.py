import pytest
import importlib.util
import pathlib
from qiskit import QuantumCircuit

# --- Dynamically import 01-shor_noiseless_demo.py ---
file_path = pathlib.Path(__file__).parent / "01-shor_noiseless_demo.py"
spec = importlib.util.spec_from_file_location("shor_module", file_path)
shor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shor_module)

# Access functions from the dynamically imported module
try_factors_from_r = shor_module.try_factors_from_r
build_shor_circuit = shor_module.build_shor_circuit

def test_try_factors_from_r_valid():
    """Test that known valid r values yield correct non-trivial factors."""
    # For N = 15 and a = 7, r = 4 is a valid order
    result = try_factors_from_r(a=7, r=4, N=15)
    assert result == (3, 5) or result == (5, 3), "Expected factors of 15 are 3 and 5."

def test_try_factors_from_r_invalid():
    """Test that invalid or odd r values return None."""
    # r is odd — should not yield factors
    assert try_factors_from_r(a=7, r=3, N=15) is None
    # gcd or result produces trivial factors
    assert try_factors_from_r(a=2, r=2, N=15) is None

def test_build_shor_circuit_structure():
    """Verify that the circuit is built correctly with expected number of qubits and classical bits."""
    qc = build_shor_circuit(a=7, N=15, t=8)
    # Expect 8 counting + 4 work qubits = 12 total, and 8 classical bits
    assert qc.num_qubits == 12
    assert qc.num_clbits == 8
    # Ensure there are measurement operations
    assert any(op[0].name == "measure" for op in qc.data)
