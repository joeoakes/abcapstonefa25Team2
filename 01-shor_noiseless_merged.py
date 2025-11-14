# Project: Breaking Crypto with Quantum Simulation
# Course: CMPSC 488/IST440W
# Author: Team 2 (Yoda)
# Date Developed: 10/21/25
# Last Date Changed: 11/14/25

# Combined noiseless Shor demo:
# - Generic-N Shor implementation (no N=15 hard-coding)
# - Interactive user run (Part 1)
# - Simple resource analysis / CSV (Part 2)
# - Circuit depth / gate-count analysis + timing benchmark (Part 3)

import math
import time
import os
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import UnitaryGate


# Global state to share the last user-chosen N and a between parts

LAST_N = None
LAST_A = None


# Generic Shor helpers


def choose_t_for_N(N: int) -> int:
    """Choose number of counting qubits based on problem size N."""
    n = math.ceil(math.log2(N))
    return max(4, n + 2)


def continued_fraction_phase_estimate(meas_value: int, t: int) -> Fraction:
    phase = meas_value / (2 ** t)
    return Fraction(phase).limit_denominator(2 ** t)


def try_order_from_measure(meas: int, t: int, a: int, N: int):
    frac = continued_fraction_phase_estimate(meas, t)
    if frac.denominator == 0:
        return None
    r = frac.denominator
    if r <= 0:
        return None
    # Try small multiples in case we only recovered k/r
    for candidate in [r, 2 * r, 3 * r, 4 * r]:
        if pow(a, candidate, N) == 1:
            return candidate
    return None


def try_factors_from_r(a: int, r: int, N: int):
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


def approximate_iqft(qc: QuantumCircuit, qubits, approximation_degree: int = 3):
    """In-place approximate inverse QFT on given qubits."""
    n = len(qubits)
    # swap
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - j - 1])
    # controlled phase + H
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


def apply_controlled_mul_a_mod_N(qc: QuantumCircuit, ctrl, work, a: int, N: int, power: int):
    """Apply |x> -> |a^power * x mod N> controlled on ctrl qubit."""
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


def build_shor_circuit_general(a: int, N: int, t: int | None = None, approximation_degree: int = 3):
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

    # work register starts in |1>
    qc.x(counting)

    # hadamards on counting register
    for i in range(counting):
        qc.h(i)

    # controlled modular exponentiation
    for k, ctrl in enumerate(range(counting)):
        power = 2 ** k
        apply_controlled_mul_a_mod_N(qc, ctrl, list(range(counting, counting + work)), a, N, power)

    approximate_iqft(qc, list(range(counting)), approximation_degree=approximation_degree)
    qc.measure(range(counting), range(counting))

    return qc, t


def shor_factor_generic(a: int, N: int, t: int | None = None, shots: int = 256, opt_level: int = 3,
                        approximation_degree: int = 3):
    if math.gcd(a, N) != 1:
        return None, None
    qc, t_eff = build_shor_circuit_general(a, N, t, approximation_degree=approximation_degree)
    sim = Aer.get_backend("qasm_simulator")
    qc_t = transpile(qc, sim, optimization_level=opt_level)
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
    return counts, factors


def plot_counts(counts, title="Shor measurement probabilities", filename="noiseless_plot.png"):
    if not counts:
        return
    total_shots = sum(counts.values())
    probs = {state: c / total_shots for state, c in counts.items()}
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    states, pvals = zip(*sorted_items)
    xs = [int(s, 2) for s in states]
    plt.figure(figsize=(8, 4))
    plt.bar(xs, pvals)
    plt.xlabel("Measurement outcome (decimal)")
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[plot] saved noiseless plot to {os.path.abspath(filename)}")



# Part 1: Interactive generic-N Shor run


def run_user_input_shor_generic():
    global LAST_N, LAST_A

    # Get N
    while True:
        try:
            N = int(input("Enter composite N to factor (e.g., 15, 21, 35, 143) (or 0 to quit): ").strip())
        except ValueError:
            print("invalid N")
            continue
        if N == 0:
            return
        if N <= 1:
            print("N must be > 1")
            continue
        break

    # Get a
    while True:
        try:
            a = int(input("Enter a (coprime to N, or 0 to re-enter N): ").strip())
        except ValueError:
            print("invalid a")
            continue
        if a == 0:
            # restart entire process
            return run_user_input_shor_generic()
        if math.gcd(a, N) != 1:
            print(f"gcd(a, N) != 1 (got {math.gcd(a, N)}); choose a different a")
            continue
        break

    LAST_N, LAST_A = N, a

    counts, factors = shor_factor_generic(a, N, shots=256)
    print("N =", N)
    print("a =", a)
    print("counts =", counts)
    print("factors =", factors)
    plot_counts(counts, title=f"Shor measurement probabilities for N={N}, a={a}",
                filename="noiseless_plot.png")


# Part 2: Simple resource analysis on the same N

def run_resource_analysis_for_last_N():
    """Run a small resource analysis for the last (N, a) and write a CSV.

    We keep this lightweight so it runs quickly but still illustrates
    success probability and runtime across different bases.
    """
    if LAST_N is None:
        print("[resource] LAST_N is not set; run Part 1 first.")
        return

    N = LAST_N
    # choose up to 3 coprime a values (could include LAST_A)
    a_values = []
    for a in range(2, N):
        if math.gcd(a, N) == 1:
            a_values.append(a)
        if len(a_values) >= 3:
            break

    trials_per_a = 5
    shots_per_trial = 128

    rows = []
    backend = Aer.get_backend("qasm_simulator")

    print(f"[resource] running resource analysis for N={N} ...")

    for a in a_values:
        for trial_idx in range(trials_per_a):
            t0 = time.perf_counter()
            counts, factors = shor_factor_generic(a, N, shots=shots_per_trial)
            dt = time.perf_counter() - t0
            success = 1 if factors is not None else 0
            rows.append([N, a, trial_idx, success, dt])

    out_name = f"results_N{N}.csv"
    import csv
    with open(out_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "a", "trial_index", "success", "time_seconds"])
        w.writerows(rows)
    print(f"[resource] wrote {out_name} with {len(rows)} rows (N={N})")

    # Simple success-rate plot per a
    success_by_a = {}
    total_by_a = {}
    for _, a, _, success, _ in rows:
        success_by_a[a] = success_by_a.get(a, 0) + success
        total_by_a[a] = total_by_a.get(a, 0) + 1
    a_list = sorted(total_by_a.keys())
    rates = [success_by_a[a] / total_by_a[a] for a in a_list]

    plt.figure(figsize=(6, 4))
    plt.bar([str(a) for a in a_list], rates)
    plt.xlabel("a")
    plt.ylabel("success rate")
    plt.title(f"Shor success rate per a for N={N}")
    plt.tight_layout()
    fname = f"resource_success_N{N}.png"
    plt.savefig(fname)
    print(f"[plot] saved resource plot to {os.path.abspath(fname)}")


# Part 3: Circuit depth / gate-count analysis for the same N


def run_depth_and_gatecount_analysis():
    if LAST_N is None or LAST_A is None:
        print("[depth] LAST_N/LAST_A not set; run Part 1 first.")
        return

    N = LAST_N
    a = LAST_A

    backend = Aer.get_backend("aer_simulator_statevector")
    qc_test, t_eff = build_shor_circuit_general(a=a, N=N)
    # no measurement here; focus on unitary part
    depth = qc_test.depth()
    width = qc_test.width()
    size = qc_test.size()
    counts_ops = qc_test.count_ops()

    print("[depth] N =", N, "a =", a)
    print("[depth] Depth:", depth)
    print("[depth] Width:", width)
    print("[depth] Size:", size)
    print("[depth] Gate counts:", counts_ops)

    # Simple bar chart: depth vs total gate count
    total_gates = sum(counts_ops.values())
    plt.figure(figsize=(6, 4))
    plt.bar(["Depth", "Total Gates"], [depth, total_gates])
    plt.ylabel("Count")
    plt.title(f"Circuit depth vs gate count for N={N}, a={a}")
    plt.tight_layout()
    fname = "depth_gatecount.png"
    plt.savefig(fname)
    print(f"[plot] saved depth/gatecount plot to {os.path.abspath(fname)}")



# Main entry: run all three parts and benchmark timing


if __name__ == "__main__":
    # Part 1: interactive Shor run
    start_time1 = time.time()
    run_user_input_shor_generic()
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1

    # If user chose to quit immediately, LAST_N will be None
    if LAST_N is None:
        print("No N chosen; skipping Parts 2 and 3.")
    else:
        # Part 2: resource analysis
        start_time2 = time.time()
        run_resource_analysis_for_last_N()
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2

        # Part 3: depth/gatecount analysis
        start_time3 = time.time()
        run_depth_and_gatecount_analysis()
        end_time3 = time.time()
        elapsed_time3 = end_time3 - start_time3

        # Benchmark timing plot for all parts
        parts = ["Part 1", "Part 2", "Part 3"]
        times = [elapsed_time1, elapsed_time2, elapsed_time3]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(parts, times, color=["tab:blue", "tab:orange", "tab:green"])
        plt.xlabel("Part")
        plt.ylabel("Elapsed Time (seconds)")
        plt.title("Benchmarking: Execution Time by Part")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar, value in zip(bars, times):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}s",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        fname = "benchmark_timing.png"
        plt.tight_layout()
        plt.savefig(fname)
        print(f"[plot] saved benchmark timing plot to {os.path.abspath(fname)}")
