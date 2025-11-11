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

from qiskit.circuit.library import QFT
from qiskit.circuit.library.standard_gates import RCCXGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation
from qiskit.synthesis.qft import synth_qft_full

import os
import time
import pylatexenc

import csv, time
from qiskit_aer import Aer
from qiskit import transpile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, contextlib

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
        #basis_gates=["cx", "u3", "id"], # target ibmq-compatible gates
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

# I provide a few types of visualization, we can choose from these as we see fit in the future

qc = build_shor_circuit(a=7, N=15, t=8)

# Only the first few layers (otherwise takes forever)

# Print summary values
print("Depth:", qc.depth())
print("Width:", qc.width())
print("Size:", qc.size())
print(qc.count_ops())

# Basic ascii circuit visualization of all circuits
#print(qc)

# Matplotlib circuit diagram
visualize_partial_circuit(qc, num_ops=50)

plt.close()
# Collect Data
# This cell runs existing Shor code many times and saves results to results.csv.

# settings
a_values = [2, 4, 7, 8, 11, 13]   # which bases to try
trials_per_a = 20                  # how many trials per base
shots_per_trial = 12               # shots each run
N = 15                             # number to factor
t = 8                              # counting precision

rows = []  # Store each row per trial: [a, trial_index, success(0/1), runtime_s]

sim = Aer.get_backend("qasm_simulator")

for a in a_values:
    for trial_idx in range(trials_per_a):
        t0 = time.perf_counter()          # start a simple timer

        # build + run using functions
        qc = build_shor_circuit(a=a, N=N, t=t)                  # make the circuit
        qc_t = transpile(qc, backend=sim, optimization_level=1) # make it compatible with backend
        result = sim.run(qc_t, shots=shots_per_trial).result()  # execute
        counts = result.get_counts()

        success = 0
        for bitstring, count in counts.items():
            meas_val = int(bitstring, 2)
            r = try_order_from_measure(meas_val, t, a, N)  # guess period r
            if r is None:
                continue
            fac = try_factors_from_r(a, r, N)        # try to get factor from r
            if fac is not None:
                success = 1
                break

        dt = time.perf_counter() - t0               # how long the trial took
        rows.append([a, trial_idx, success, dt])    # save one row

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a", "trial_index", "success", "runtime_s"])
    w.writerows(rows)                             # data row

print("Wrote results.csv with", len(rows), "rows")
end_time2 = time.time()
elapsed_time2 = end_time2 - start_time2
# === PART 3 (Thomas) Graphs Comparing Noise (Baseline)===

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
        bits = bitstring.replace(" ", "")  # remove spaces if any
        trimmed = bits[-8:]                # keep the last 8 bits only
        trimmed_counts[trimmed] = trimmed_counts.get(trimmed, 0) + count
    counts = trimmed_counts
    print("trimmed counts =", counts)
    if not counts:
        import pandas as pd
        df = pd.read_csv("results.csv")
        success_counts = df.groupby("a")["success"].sum().to_dict()
        counts = {f"a={k}": v for k, v in success_counts.items()}
        print("Loaded counts from results.csv:", counts)

    if counts:  #only plot if valid data
        total_shots = sum(counts.values())#Compute the total number of measurements taken
        probabilities = {state: count / total_shots for state, count in counts.items()}#Convert counts into probabilities (divide each count by total shots)

        N_show = 10#Choose how many of the most frequent results to display on the chart(usally only shows 4-6 but 10 just in case)
         #Sort bitstrings by probability (highest first) and keep only the top N \/
        top_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:N_show])

        plt.figure(figsize=(6,4))#Create a new window and set its size

        bars = plt.bar(top_probs.keys(), top_probs.values(), color='blue')#Draw a bar chart of bitstring probabilities
        plt.title(f"Top Measurement Outcomes for a={a} (No Noise)") #title indicating which base (a=7)(or whatever we use) and that it’s a no noise baseline measurement 
        plt.xlabel("Measured Bitstring") #label horizontal axis
        plt.ylabel("Probability") #label vertical axis
        plt.xticks(rotation=45)#Rotate x-axis labels for readability
        plt.tight_layout() #Automatically adjust layout so labels and title fit neatly

        for bar, val in zip(bars, top_probs.values()):#Loop through each bar and print its probability percentage above the bar
            plt.text(
                bar.get_x() + bar.get_width()/2, #horizontally center text above bar
                bar.get_height() + 0.01,#lace text slightly above bar height
                f"{val*100:.1f}%",#show value as a percentage (1 decimal)
                ha='center', va='bottom', fontsize=10#center alignment and font size
            )
    print("Figures available:", plt.get_fignums())
    plt.savefig("noiseless_plot.png")#display chart
    print("saved noiseless_plot.png in", os.getcwd())
 
end_time3 = time.time()
elapsed_time3 = end_time3 - start_time3
parts = ['Part 2', 'Part 3']
times = [elapsed_time2, elapsed_time3]

# Create a bar chart
parts = ['Part 2', 'Part 3']
times = [elapsed_time2, elapsed_time3]

plt.figure(figsize=(6,4))
bars = plt.bar(parts, times)

plt.xlabel('Code Section')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Benchmarking: Execution Time by Part')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on each bar
for bar, value in zip(bars, times):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
        value,                              # y position (top of bar)
        f"{value:.3f}s",                    # label text (rounded to 3 decimals)
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

plt.savefig('benchmark_timing.png')
print("saved benchmark_timing.png in", os.getcwd())
