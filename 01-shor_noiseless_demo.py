# Project:
# Purpose Details:
# Course: CMPSC 488/IST440W
# Author: Team Yoda
# Date Developed: 10/21/25
# Last Date Changed: 10/22/25
# Revision:

 
!pip install qiskit
!pip install qiskit qiskit-aer
!pip install matplotlib
!pip install pylatexenc

import math
import pylatexenc
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

def cswap_decomp(qc, c, a, b):
    qc.ccx(a, c, b)
    qc.ccx(b, c, a)
    qc.ccx(a, c, b)

def iqft_in_place(qc, qubits):
    n = len(qubits)
    for j in range(n//2):
        qc.swap(qubits[j], qubits[n-1-j])
    for j in range(n-1, -1, -1):
        for k in range(j+1, n):
            qc.cp(-math.pi/(2**(k-j)), qubits[k], qubits[j])
        qc.h(qubits[j])

def apply_controlled_mul_a_mod_15(qc, ctrl, work, a, power):
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
            qc.cx(ctrl, work[0]); qc.cx(ctrl, work[1]); qc.cx(ctrl, work[2]); qc.cx(ctrl, work[3])

def controlled_U(qc, controls, work, a):
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

def build_shor_circuit(a, N=15, t=8):
    counting = t
    work = 4
    qc = QuantumCircuit(counting + work, counting)
    qc.x(counting)
    for i in range(counting):
        qc.h(i)
    controlled_U(qc, range(counting), list(range(counting, counting + work)), a)
    iqft_in_place(qc, list(range(counting)))
    qc.measure(range(counting), range(counting))
    return qc

def shor_factor_demo(a, N=15, t=8, shots=12):
    if math.gcd(a, N) != 1:
        print(f"gcd({a},{N}) != 1:", math.gcd(a, N)); return
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

# === PART 2 (Marco) --- Visualizing Ciruits ===

# function to visualize prefix of circuit with matplotlib (can change num_ops for more/less circuits)
def visualize_partial_circuit(qc, num_ops=50):
  partial = QuantumCircuit(qc.num_qubits, qc.num_clbits)
  for instr, qargs, cargs in qc.data[:num_ops]:
    partial.append(instr, qargs, cargs)
  return partial.draw('mpl')

# I provide a few types of visualization, we can choose from these as we see fit in the future

if __name__ == "__main__":
    for a in [2, 7, 8, 11, 13, 4]:
        print("\n" + "="*60)
        print(f"Running Shor for N=15, a={a}")
        shor_factor_demo(a=a, N=15, t=8, shots=12)

  qc = build_shor_circuit(a=7, N=15, t=8)

  # only the first few layers (otherwise takes forever)
  
  # print summary values
  print("Depth:", qc.depth())
  print("Width:", qc.width())
  print("Size:", qc.size())
  print(qc.count_ops())
  
  # basic ascii circuit visualization of all circuits
  #print(qc)
  
  # matplotlib circuit diagram
  visualize_partial_circuit(qc, num_ops=50)

# === (Thomas) Graphs Comparing Noise (Baseline)===
#comments for every line since i will be absent and someone will have to explain code...
import matplotlib.pyplot as plt
import io, contextlib  

if __name__ == "__main__": #only runs if above is running right
    a = 7 #the base value a used is Shor's alg
    buf = io.StringIO()

    with contextlib.redirect_stdout(buf): #makes sure things (like “Candidate period r = …”) doesnt showup
        counts = shor_factor_demo(a=a, N=15, t=8, shots=1024) ##Run the Shor algorithm demo for given parameters (a=7, N=15, t=8) with 1024 shots
         #Returns a dictionary of measurement outcomes (bitstrings) and how often they occurred

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

        plt.show()#display chart
