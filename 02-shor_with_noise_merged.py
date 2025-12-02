''' # required dependancies
!pip install qiskit --quiet
!pip install qiskit_aer --quiet
!pip install qiskit_ibm_runtime --quiet
'''
# === PART 3 (Marcos) --- Introducing Noise ===
import math
from fractions import Fraction
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, errors
from qiskit.circuit.library import UnitaryGate

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def choose_t_for_N(N):
    n = math.ceil(math.log2(N))
    return max(4, n + 2)



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
    for candidate in [r, 2 * r, 3 * r, 4 * r]:
        if pow(a, candidate, N) == 1:
            return candidate
    return None


def try_factors_from_r(a, r, N):
    if r is None or r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x in [1, N - 1, 0]:
        return None
    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)
    if p * q == N and p not in [1, N] and q not in [1, N]:
        return (p, q)
    return None


def approximate_iqft(qc, qubits, approximation_degree=3):
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - j - 1])
    for j in range(n):
        for k in range(j):
            if j - k <= approximation_degree:
                qc.cp(-math.pi / (2 ** (j - k)), qubits[j], qubits[k])
        qc.h(qubits[j])


def _build_perm_matrix_from_mapping(mapping, dim):
    mat = np.zeros((dim, dim), dtype=complex)
    for x, y in mapping.items():
        mat[y, x] = 1.0
    return mat


def apply_controlled_mul_a_mod_N(qc, ctrl, work, a, N, power):
    n_work = len(work)
    dim = 2 ** n_work
    if N > dim:
        raise ValueError(f"work register too small for N={N} (have {n_work} qubits)")
    a_k = pow(a, power, N)
    mapping = {}
    for x in range(dim):
        if x < N:
            y = (a_k * x) % N
        else:
            y = x
        mapping[x] = y
    mat = _build_perm_matrix_from_mapping(mapping, dim)
    gate = UnitaryGate(mat, label=f"mul_{a_k}_mod_{N}")
    cgate = gate.control(1)
    qc.append(cgate, [ctrl] + list(work))


def build_shor_circuit_general(a, N, t=None, approximation_degree=3):
    if t is None:
        t = choose_t_for_N(N)
    counting = t
    work = math.ceil(math.log2(N))
    dim = 2 ** work
    if N > dim:
        raise ValueError(f"work register too small for N={N} (have {work} qubits)")

    qc = QuantumCircuit(counting + work)
    creg = ClassicalRegister(t, "c")
    qc.add_register(creg)

    qc.x(counting)

    for i in range(counting):
        qc.h(i)

    for k, ctrl in enumerate(range(counting)):
        power = 2 ** k
        apply_controlled_mul_a_mod_N(qc, ctrl, list(range(counting, counting + work)), a, N, power)

    approximate_iqft(qc, list(range(counting)), approximation_degree=approximation_degree)
    qc.measure(range(counting), range(counting))

    return qc, t


def create_simple_noise_model():
    noise_model = NoiseModel()
    p_depol = 0.01
    p_readout = 0.02
    noise_model.add_all_qubit_quantum_error(errors.depolarizing_error(p_depol, 1), ['u1', 'u2', 'u3', 'x', 'h'])
    noise_model.add_all_qubit_readout_error(errors.readout_error.ReadoutError([[1 - p_readout, p_readout],
                                                                                [p_readout, 1 - p_readout]]))
    return noise_model


def shor_with_noise_generic(a, N, t=None, shots=256, noisy=True):
    if math.gcd(a, N) != 1:
        return None, None, None
    qc, t_eff = build_shor_circuit_general(a, N, t)
    if noisy:
        noise_model = create_simple_noise_model()
        sim = Aer.get_backend("aer_simulator")
        qc_t = transpile(qc, sim)
        result = sim.run(qc_t, shots=shots, noise_model=noise_model).result()
    else:
        sim = Aer.get_backend("qasm_simulator")
        qc_t = transpile(qc, sim)
        result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    factors = None
    for bitstring, count in counts.items():
        meas_val = int(bitstring, 2)
        r = try_order_from_measure(meas_val, t_eff, a, N)
        if r is None:
            continue
        cand = try_factors_from_r(a, r, N)
        if cand is not None:
            factors = cand
            break
    return counts, factors, {"N": N, "a": a, "noisy": noisy}


def plot_counts(counts, title="Noisy Shor measurement probabilities", filename="noisy_plot.png"):
    if not counts:
        return
    total_shots = sum(counts.values())
    probs = {state: c / total_shots for state, c in counts.items()}
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    states, pvals = zip(*sorted_items)
   # xs = [int(s, 2) for s in states]
    plt.figure(figsize=(8, 4))
    wcag_colors = ["#003366", "#054A29", "#7A0000", "#2E0057", "#004F4F", "#4A2900"]
    plt.bar(states, pvals, color=wcag_colors[:len(states)], edgecolor="black")
    plt.xlabel("Measured Bitstring")
    plt.ylabel("Probability")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[plot] saved noisy plot to {os.path.abspath(filename)}")

    


def run_user_input_shor_with_noise():
    while True:
        try:
            N = int(input("Enter composite N to factor with noise (e.g., 15, 21, 35, 143) (or 0 to quit): ").strip())
        except ValueError:
            print("invalid N")
            continue
        if N == 0:
            return
        if N <= 1:
            print("N must be > 1")
            continue
        break

    while True:
        try:
            a = int(input("Enter a (coprime to N, or 0 to re-enter N): ").strip())
        except ValueError:
            print("invalid a")
            continue
        if a == 0:
            return run_user_input_shor_with_noise()
        if math.gcd(a, N) != 1:
            print(f"gcd(a, N) != 1 (got {math.gcd(a, N)}); choose a different a")
            continue
        break

    noisy_str = input("Add noise? [y/n] (default y): ").strip().lower()
    noisy = (noisy_str in ("", "y", "yes"))

    counts, factors, info = shor_with_noise_generic(a, N, noisy=noisy)
    print("N =", info["N"])
    print("a =", info["a"])
    print("noisy =", info["noisy"])
    print("counts =", counts)
    print("factors =", factors)
    suffix = "noisy" if noisy else "ideal"
    plot_counts(counts, title=f"Noisy Shor probabilities for N={N}, a={a}, noisy={noisy}", filename=f"noisy_plot_{suffix}.png")


if __name__ == "__main__":
    run_user_input_shor_with_noise()


#  Benchmark Timing for Noisy vs Ideal
import time
import contextlib
import io

def run_default_noisy_benchmark():
    """Run a small benchmark for N=15, a=7 (can be changed) comparing
    ideal vs noisy Shor runs and save:
      - noisy_plot.png            (distribution for noisy case)
      - 02benchmark_timing.png    (bar chart of Part 1 vs Part 2 time)
    """
    N = 15
    a = 7

    # Part 1: run ideal + noisy once each and time them together
    start_time1 = time.time()
    # ideal run
    _counts_ideal, _factors_ideal, _info_ideal = shor_with_noise_generic(a, N, noisy=False)
    # noisy run 
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        counts_noisy, factors_noisy, info_noisy = shor_with_noise_generic(a, N, noisy=True)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    # Part 2: plotting only (from noisy counts)
    start_time2 = time.time()
    if counts_noisy:
        # reuse existing plotting helper but force a consistent filename
        plot_counts(
            counts_noisy,
            title=f"Top Measurement Outcomes for a={a}, N={N} (With Noise)",
            filename="noisy_plot.png",
        )
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2

    # Create benchmark timing bar chart
    parts = ['Part 1', 'Part 2', 'Part 3']
    times = [elapsed_time1, elapsed_time2, elapsed_time3]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(parts, times)

    plt.xlabel('Code Section')
    plt.ylabel('Elapsed Time (seconds)')
    plt.title('Noisy Shor Runtime Part 1 Only')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, value in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}s",
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig('02benchmark_timing.png')
    print("saved noisy_plot.png and 02benchmark_timing.png in", os.getcwd())


if __name__ == "__main__":
    # Main interactive entrypoint
    run_user_input_shor_with_noise()
    run_default_noisy_benchmark()
