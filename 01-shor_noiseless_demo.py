# Project: Breaking Crypto with Quantum Simulation
# Course: CMPSC 488/IST440W
# Author: Team 2 (Yoda)
# Date Developed: 10/21/25
# Last Date Changed: 10/23/25

import math
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

import os

import pylatexenc

import csv, time
from qiskit_aer import Aer
from qiskit import transpile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, contextlib

# === PART 1 (Martin) --- Shor's Algorithm ===
def cswap_decomp(qc, c, a, b):
    # Controlled swap between qubits a and b if control qubit c = 1
    qc.ccx(a, c, b)
    qc.ccx(b, c, a)
    qc.ccx(a, c, b)

def iqft_in_place(qc, qubits):
    # Inverse Quantum Fourier Transform on given qubits
    n = len(qubits)
    for j in range(n//2):
        qc.swap(qubits[j], qubits[n-1-j])
    for j in range(n-1, -1, -1):
        for k in range(j+1, n):
            qc.cp(-math.pi/(2**(k-j)), qubits[k], qubits[j])
        qc.h(qubits[j])

def apply_controlled_mul_a_mod_15(qc, ctrl, work, a, power):
    # Controlled modular multiplication by 'a' mod 15
    for _ in range(power):
        if a in [2, 13]:
            cswap_decomp(qc, ctrl, work[2], work[3])
            cswap_decomp(qc, ctrl, work[1], work[2])
            cswap_decomp(qc, ctrl, work[0], work[1])
        if a in [7, 8]:
            cswap_decomp(qc, ctrl, work[0], work[1])
            cswap_decomp(qc, ctrl, work[1], work[2])
            cswap_decomp(qc, ctrl, work[2], work[3])
        if a in [4, 11]:
            cswap_decomp(qc, ctrl, work[1], work[3])
            cswap_decomp(qc, ctrl, work[0], work[2])
        if a in [7, 11, 13]:
            qc.cx(ctrl, work[0])
            qc.cx(ctrl, work[1])
            qc.cx(ctrl, work[2])
            qc.cx(ctrl, work[3])

def controlled_U(qc, controls, work, a):
    # Apply controlled powers of U (a^2^k mod 15)
    for k, ctrl in enumerate(controls):
        apply_controlled_mul_a_mod_15(qc, ctrl, work, a, 2**k)

def continued_fraction_phase_estimate(meas_value, t):
    # Convert measurement to fraction for period estimation
    phase = meas_value / (2 ** t)
    return Fraction(phase).limit_denominator(2 ** t)

def try_order_from_measure(meas, t, a, N):
    # Try to find the order r from the measurement result
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
    # Try to compute non-trivial factors using r
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

def build_shor_circuit(a, N=15, t=8):
    # Build Shor's order-finding circuit
    counting = t
    work = 4
    qc = QuantumCircuit(counting + work, counting)
    qc.x(counting)  # set work register to |1>
    for i in range(counting):
        qc.h(i)  # put counting qubits in superposition
    controlled_U(qc, range(counting), list(range(counting, counting + work)), a)
    iqft_in_place(qc, list(range(counting)))  # apply inverse QFT
    qc.measure(range(counting), range(counting))
    return qc

def shor_factor_demo(a, N=15, t=8, shots=12):
    # Run simulation and extract factors
    if math.gcd(a, N) != 1:
        print(f"gcd({a},{N}) != 1:", math.gcd(a, N))
        return
    qc = build_shor_circuit(a, N, t)
    sim = Aer.get_backend("qasm_simulator")
    qc_t = transpile(qc, backend=sim, optimization_level=1)
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
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
            return
        else:
            print("No factors from this r.")
    print("No non-trivial factors found.")
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "trial_index", "success", "runtime_s"])
        w.writerows(rows)

    print("Wrote results.csv with", len(rows), "rows")

    # Convert rows into counts dictionary
    counts = {}
    for a, trial_index, success, runtime_s in rows:
        key = f"a={a}_trial={trial_index}_success={success}"
        counts[key] = counts.get(key, 0) + 1

    return counts
    return counts
# === PART 2 (Marco) --- Visualizing Ciruits ===
# Function to visualize prefix of circuit with matplotlib (can change num_ops for more/less circuits)
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
# === PART 3 (Thomas) Graphs Comparing Noise (Baseline)===
if __name__ == "__main__": #only runs if above is running right
    a = 7 #the base value a used is Shor's alg
    buf = io.StringIO()

    from qiskit_aer import Aer
    from qiskit import transpile

    with contextlib.redirect_stdout(buf):
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
