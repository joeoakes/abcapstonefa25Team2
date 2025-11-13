# === PART 4 (Vasu) --- Analyze Resource Usage ===
# Collect Data
# This cell runs existing Shor code many times and saves results to results.csv.

import math
import csv
import time
from fractions import Fraction

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
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


def build_shor_circuit_general(a, N, t=None):
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

    # simple exact inverse QFT
    n = counting
    for j in range(n // 2):
        qc.swap(j, n - j - 1)
    for j in range(n):
        for k in range(j):
            qc.cp(-math.pi / (2 ** (j - k)), j, k)
        qc.h(j)

    qc.measure(range(counting), range(counting))
    return qc, t


def shor_factor_generic(a, N, t=None, shots=1024, opt_level=1):
    if math.gcd(a, N) != 1:
        return None, None, None
    qc, t_eff = build_shor_circuit_general(a, N, t)
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
    return counts, factors, t_eff


def run_resource_analysis_generic():
    try:
        N = int(input("Enter composite N to factor for resource analysis: ").strip())
    except ValueError:
        print("invalid N")
        return
    if N <= 1:
        print("N must be > 1")
        return

    try:
        max_a = int(input("Max number of diffe21rent a values to test (default 5): ").strip() or "5")
    except ValueError:
        max_a = 5

    try:
        trials_per_a = int(input("Trials per a (default 5): ").strip() or "5")
    except ValueError:
        trials_per_a = 5

    try:
        shots_per_trial = int(input("Shots per trial (default 128): ").strip() or "128")
    except ValueError:
        shots_per_trial = 128

    a_values = []
    for a in range(2, N):
        if math.gcd(a, N) == 1:
            a_values.append(a)
        if len(a_values) >= 3:
            break

    if not a_values:
        print("no coprime a values found for N")
        return

    rows = []
    for a in a_values:
        for trial_idx in range(trials_per_a):
            t0 = time.perf_counter()
            counts, factors, t_eff = shor_factor_generic(a, N, shots=shots_per_trial)
            dt = time.perf_counter() - t0
            success = 1 if factors is not None else 0
            rows.append([N, a, trial_idx, success, dt, t_eff])

    out_name = f"results_N{N}.csv"
    with open(out_name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "a", "trial_index", "success", "time_seconds", "t_eff"])
        w.writerows(rows)
    print("saved generic resource data to", out_name)

    # simple success-rate bar plot
    successes = {}
    totals = {}
    for row in rows:
        _, a, _, success, _, _ = row
        successes[a] = successes.get(a, 0) + success
        totals[a] = totals.get(a, 0) + 1
    a_list = sorted(successes.keys())
    rates = [successes[a] / totals[a] for a in a_list]
    plt.figure(figsize=(6, 4))
    plt.bar([str(a) for a in a_list], rates)
    plt.xlabel("a")
    plt.ylabel("success rate")
    plt.title(f"Shor success rate per a for N={N}")
    plt.tight_layout()
    plt.savefig(f"resource_success_N{N}.png")
    print(f"[plot] saved resource plot to {os.path.abspath(f'resource_success_N{N}.png')}")


if __name__ == "__main__":
    run_resource_analysis_generic()
