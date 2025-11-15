# Project: Breaking Crypto with Quantum Simulation
# Course: CMPSC 488/IST440W
# Author: Team 2 (Yoda)
# Date Developed: 10/21/25
# Last Date Changed: 10/23/25

# marco's shor's optimizations added 11/11/25

import math
from fractions import Fraction
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, errors  # <-- added for noise support

from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import RCCXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit.synthesis.qft import synth_qft_full

import os
import time
import pylatexenc

import csv, time
from qiskit import transpile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, contextlib


# === GPU auto-detection helper ===
# Auto targets gpu instead of cpu (if available)
from qiskit_aer import AerSimulator

def get_best_backend():
    try:
        # Try GPU-enabled simulator if available
        gpu_backend = AerSimulator(method='statevector', device='GPU')
        print("✅ Using GPU-accelerated AerSimulator")
        return gpu_backend
    except Exception as e:
        # Fallback to standard CPU simulator
        print("⚙️ GPU not available, using CPU simulator")
        return Aer.get_backend("aer_simulator")
        
        

# === PART 1 (Martin) --- Shor's Algorithm ===
start_time1 = time.time()
# removed old cswap_decomp, replaced with native cswap

# This avoids 'InstructionSet' errors while achieving the same effect:
# fewer gates due to truncated rotations and optimized synthesis.

def approximate_iqft(qc, qubits, approximation_degree=3):
    """Approximate inverse QFT using Qiskit synthesis backend"""
    iqft_circ = synth_qft_full(num_qubits=len(qubits), inverse=True, approximation_degree=approximation_degree)
    qc.append(iqft_circ, qubits)

def apply_controlled_mul_a_mod_15(qc, ctrl, work, a, power):
  # optimized modular multiplication circuit for n=15
  # uses lookup table instread of repeated if statements

  # use qiskit.circuit.library.arithmetic for larger!

  # lookup tables for swap networks
  swap_patterns = {
      2: [(2, 3), (1, 2), (0, 1)],
        13: [(2, 3), (1, 2), (0, 1)],
        7: [(0, 1), (1, 2), (2, 3)],
        8: [(0, 1), (1, 2), (2, 3)],
        4: [(1, 3), (0, 2)],
        11: [(1, 3), (0, 2)],
    }
  xor_all = {7, 11, 13} # a values that need full bit flips

  swaps = swap_patterns.get(a, [])
  for _ in range(power):
        for pair in swaps:
          qc.cswap(ctrl, work[pair[0]], work[pair[1]])
        if a in xor_all:
          for i in range(4):
            qc.cx(ctrl, work[i])

def controlled_U(qc, controls, work, a):
  # apply modular exponentiation block using powers of a
    for k, ctrl in enumerate(controls):
        apply_controlled_mul_a_mod_15(qc, ctrl, work, a, 2**k)

def continued_fraction_phase_estimate(meas_value, t):
    phase = meas_value / (2 ** t)
    return Fraction(phase).limit_denominator(2 ** t)

def try_order_from_measure(meas, t, a, N):
    frac = continued_fraction_phase_estimate(meas, t)
    if frac.denominator == 0:
        return None
    r = frac.denominator
    if r <= 0:
        return None
    for candidate in [r, 2*r, 3*r, 4*r]:
        if pow(a, candidate, N) == 1:
            return candidate
    return None

def try_factors_from_r(a, r, N):
    if r is None or r % 2 == 1:
        return None
    x = pow(a, r // 2, N)
    if x in [1, N-1, 0]:
        return None
    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)
    if p * q == N and p not in [1, N] and q not in [1, N]:
        return (p, q)
    return None

def build_shor_circuit(a, N=15, t=8, approximation_degree=3):
    counting = t
    work = 4
    qc = QuantumCircuit(counting + work)

    creg = ClassicalRegister(t, "c")
    qc.add_register(creg)

    # Step 1: Initialize |1> state in the work register
    qc.x(counting)

    # Step 2: Hadamard on counting register
    for i in range(counting):
        qc.h(i)

    # Step 3: Apply controlled modular exponentiation
    controlled_U(qc, range(counting), list(range(counting, counting + work)), a)

    # Step 4: Apply inverse QFT (approximate)
    # replaced iqft_in_place with semi-classical iqft
    approximate_iqft(qc, list(range(counting)), approximation_degree=approximation_degree)

    # Step 5: Measure counting register
    qc.measure(range(counting), range(counting))

    return qc

def shor_factor_demo(a, N=15, t=8, shots=12, opt_level=3):
    if math.gcd(a, N) != 1:
        print(f"gcd({a},{N}) != 1:", math.gcd(a, N))
        return

    qc = build_shor_circuit(a, N, t)
    sim = Aer.get_backend("qasm_simulator")


    pm = PassManager([Optimize1qGates(), CommutativeCancellation()])

    # run pass manager for early optimizations (1q merges, cancellations)
    # returns new qc in place
    qc_after_pm = pm.run(qc)

    # High-level transpilation optimization
    qc_t = transpile(
        qc_after_pm,
        backend=sim,
        optimization_level=opt_level,
        layout_method="sabre",
        routing_method="sabre",
    )

    # --- Diagnostics ---
    print(f"Qubits: {qc_t.num_qubits}, Depth: {qc_t.depth()}")
    print("Gate counts:", qc_t.count_ops())

    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Postprocess results
    for bitstring, count in counts.items():
        meas_val = int(bitstring, 2)
        r = try_order_from_measure(meas_val, t, a, N)
        if r is None:
            continue
        print(f"Candidate period r = {r}")
        factors = try_factors_from_r(a, r, N)
        if factors is not None:
            p, q = factors
            print(f"SUCCESS: {p} × {q} = {N}")
            return counts
        else:
            print("No factors from this r.")
    print("No non-trivial factors found.")
    return counts
    return counts
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
# === PART 2 (Marco) --- Visualizing Ciruits ===
# Function to visualize prefix of circuit with matplotlib (can change num_ops for more/less circuits)
start_time2 = time.time()
def visualize_partial_circuit(qc, num_ops=50):
  partial = QuantumCircuit(qc.num_qubits, qc.num_clbits)
  for instr, qargs, cargs in qc.data[:num_ops]:
    partial.append(instr, qargs, cargs)
  return partial.draw('mpl')

qc = build_shor_circuit(a=7, N=15, t=8)
print("Depth:", qc.depth())
print("Width:", qc.width())
print("Size:", qc.size())
print(qc.count_ops())
visualize_partial_circuit(qc, num_ops=50)
plt.close()
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2

# === PART 3 (Thomas/Marcos) Graphs Comparing Noise (Baseline)===

start_time3 = time.time()
if __name__ == "__main__": #only runs if above is running right
    a = 7 #the base value a used is Shor's alg

    backend = Aer.get_backend("aer_simulator_statevector")
    qc = build_shor_circuit(a=a, N=15, t=8)
    qc.measure_all()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=2048).result()
    counts = result.get_counts()
    print("counts =", counts)

    trimmed_counts = {}
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        trimmed = bits[-8:]
        trimmed_counts[trimmed] = trimmed_counts.get(trimmed, 0) + count
    counts = trimmed_counts
    print("trimmed counts =", counts)

    if counts:
        total_shots = sum(counts.values())
        probabilities = {state: count / total_shots for state, count in counts.items()}
        N_show = 10
        top_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:N_show])

        plt.figure(figsize=(6,4))
        bars = plt.bar(top_probs.keys(), top_probs.values(), color='blue')
        plt.title(f"Top Measurement Outcomes for a={a} (No Noise)")
        plt.xlabel("Measured Bitstring")
        plt.ylabel("Probability")
        plt.xticks(rotation=45)
        plt.tight_layout()
        for bar, val in zip(bars, top_probs.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val*100:.1f}%", ha='center', va='bottom', fontsize=10)
    print("Figures available:", plt.get_fignums())
    plt.savefig("noiseless_plot.png")
    print("saved noiseless_plot.png in", os.getcwd())

    # === Depolarizing Noise ===
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(errors.depolarizing_error(0.02, 1), ['x', 'h', 'u3'])
    noise_model.add_all_qubit_quantum_error(errors.depolarizing_error(0.04, 2), ['cx'])

    noisy_backend = AerSimulator(method='density_matrix', device='GPU')
    tqc_noisy = transpile(qc, noisy_backend)
    result_noisy = noisy_backend.run(tqc_noisy, shots=2048, noise_model=noise_model).result()
    dep_counts = result_noisy.get_counts()
    plt.figure(figsize=(6,4))
    plt.bar(dep_counts.keys(), [v/sum(dep_counts.values()) for v in dep_counts.values()], color='red')
    plt.title("Measurement Outcomes with Depolarizing Noise")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("depolarizing_plot.png")
    print("saved depolarizing_plot.png in", os.getcwd())

    # === Amplitude Damping Noise ===
    amp_noise = NoiseModel()
    amp_noise.add_all_qubit_quantum_error(errors.amplitude_damping_error(0.05), ['x', 'h', 'u3'])
    two_qubit_amp_error = errors.amplitude_damping_error(0.1).tensor(errors.amplitude_damping_error(0.1))
    amp_noise.add_all_qubit_quantum_error(two_qubit_amp_error, ['cx'])

    result_amp = noisy_backend.run(tqc_noisy, shots=2048, noise_model=amp_noise).result()
    amp_counts = result_amp.get_counts()
    plt.figure(figsize=(6,4))
    plt.bar(amp_counts.keys(), [v/sum(amp_counts.values()) for v in amp_counts.values()], color='orange')
    plt.title("Measurement Outcomes with Amplitude Damping Noise")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("amplitude_plot.png")
    print("saved amplitude_plot.png in", os.getcwd())

    # === Phase Damping Noise ===
    phase_noise = NoiseModel()
    phase_noise.add_all_qubit_quantum_error(errors.phase_damping_error(0.05), ['x', 'h', 'u3'])
    two_qubit_phase_error = errors.phase_damping_error(0.1).tensor(errors.phase_damping_error(0.1))
    phase_noise.add_all_qubit_quantum_error(two_qubit_phase_error, ['cx'])

    result_phase = noisy_backend.run(tqc_noisy, shots=2048, noise_model=phase_noise).result()
    phase_counts = result_phase.get_counts()
    plt.figure(figsize=(6,4))
    plt.bar(phase_counts.keys(), [v/sum(phase_counts.values()) for v in phase_counts.values()], color='green')
    plt.title("Measurement Outcomes with Phase Damping Noise")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("phase_plot.png")
    print("saved phase_plot.png in", os.getcwd())

end_time3 = time.time()
elapsed_time3 = end_time3 - start_time3
parts = ['Part 2', 'Part 3']
times = [elapsed_time2, elapsed_time3]

plt.figure(figsize=(6,4))
bars = plt.bar(parts, times)
plt.xlabel('Code Section')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Benchmarking: Execution Time by Part')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar, value in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.savefig('benchmark_timing.png')
print("saved benchmark_timing.png in", os.getcwd())
