# shor_semiclassical.py
# Project: Breaking Crypto with Quantum Simulation
# Course: CMPSC 488/IST440W
# Author: Team 2 (Yoda)
# Date Developed: 10/21/25
# Last Date Changed: 12/02/25

import math
import time
import os
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dependencies
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Reset, UnitaryGate

# ==============================================================================
#  CORE LOGIC: MANUAL MATRIX
# ==============================================================================

def build_full_controlled_matrix(a, N, power, n_work):
    dim_work = 2 ** n_work
    dim_total = 2 * dim_work 
    # complex64 is critical for GPU speed (Single Precision)
    mat = np.eye(dim_total, dtype=np.complex64)
    val_pow = pow(a, power, N)
    
    mat[dim_work:, dim_work:] = 0
    x_indices = np.arange(dim_work)
    y_indices = np.where(x_indices < N, (x_indices * val_pow) % N, x_indices)
    mat[dim_work + y_indices, dim_work + x_indices] = 1.0
    return mat

def build_shor_circuit_semiclassical(a, N):
    n_bits = math.ceil(math.log2(N))
    t = 2 * n_bits 
    n_work = n_bits
    
    qc = QuantumCircuit(1 + n_work)
    creg = ClassicalRegister(t, "c")
    qc.add_register(creg)

    q_ctrl = 0
    q_work = list(range(1, 1 + n_work))
    qc.x(q_work[0]) 

    print(f"   [Circuit] Building Circuit (t={t})...", end="", flush=True)
    
    for i in range(t):
        qc.append(Reset(), [q_ctrl])
        qc.h(q_ctrl)
        
        current_power = 2**(t - 1 - i)
        
        # Build Matrix (Instant)
        mat = build_full_controlled_matrix(a, N, current_power, n_work)
        # check_input=False prevents CPU validation hang
        gate = UnitaryGate(mat, label=f"CMul_{current_power}", check_input=False)
        
        # Append (No .control() call!)
        qc.append(gate, q_work + [q_ctrl])
        
        for j in range(i):
            angle = -math.pi / (2**(i-j))
            qc.p(angle, q_ctrl).c_if(creg[j], 1)
            
        qc.h(q_ctrl)
        qc.measure(q_ctrl, creg[i])

    print(" Done.")
    return qc, t

# ==============================================================================
#  RUNNER
# ==============================================================================

def get_factors_from_r(a, r, N):
    if r is None or r == 0: return None
    if r % 2 != 0: return None
    x = pow(a, r // 2, N)
    if x == 1 or x == N - 1: return None
    p = math.gcd(x - 1, N)
    q = math.gcd(x + 1, N)
    if p * q == N and p not in [1, N] and q not in [1, N]: return (p, q)
    return None

def get_order_from_meas(meas, t, a, N):
    phase = meas / (2 ** t)
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    return r

def run_simulation_streaming(N, a):
    # 1. Pre-Check
    if math.gcd(a, N) != 1: 
        print(f"   [Check] gcd({a}, {N}) != 1. Trivial factor.")
        return {}, (math.gcd(a,N), N//math.gcd(a,N)), 0.0

    # 2. Build
    qc, t_eff = build_shor_circuit_semiclassical(a, N)

    # Visualization (Marco)
    def safe_visualize(circ, name, folder="qc_output"):
        try:
            os.makedirs(folder, exist_ok=True)
            fig = circ.draw(output="mpl")
            path = os.path.join(folder, name)
            fig.savefig(path, dpi=200)
            print(f"   [Viz] Saved: {path}")
        except Exception as e:
            print(f"   [Viz] Failed ({name}): {e}")

    print("   [Viz] Generating circuit visualizations...")

    # Raw circuit
    safe_visualize(qc, "raw_circuit.png")

    # Transpiled visualization
    try:
        sim_temp = AerSimulator(method='statevector')
        target_basis = sim_temp.configuration().basis_gates + ['unitary']
        qc_t_temp = transpile(qc, sim_temp, basis_gates=target_basis, optimization_level=0)
        safe_visualize(qc_t_temp, "transpiled_circuit.png")
    except Exception as e:
        print(f"   [Viz] Skipped transpiled visualization: {e}")

    # DAG
    try:
        from qiskit.visualization import dag_drawer
        from qiskit.converters import circuit_to_dag
        dag = circuit_to_dag(qc)
        fig = dag_drawer(dag, output="mpl")
        fig.savefig(os.path.join("qc_output", "circuit_dag.png"), dpi=200)
        print("   [Viz] Saved: qc_output/circuit_dag.png")
    except Exception as e:
        print(f"   [Viz] Skipped DAG visualization: {e}")
    
    # 3. Simulator (Force GPU)
    sim = AerSimulator(method='statevector')
    try:
        sim.set_options(device='GPU')
        sim.set_options(precision='single') # Fast Mode
        print("   [System] GPU Locked (RTX 4090).")
    except:
        print("   [Error] GPU init failed.")
        return {}, None, 0.0

    # 4. Transpile (Level 0)
    print(f"   [Sim] Transpiling (Direct)...", end="", flush=True)
    target_basis = sim.configuration().basis_gates + ['unitary']
    qc_t = transpile(qc, sim, basis_gates=target_basis, optimization_level=0)
    print(" Done.")

    print(f"   [Sim] Starting Stream (Max 20 attempts)...")
    print("         NOTE: Attempt 1 will take ~10-20s to compile GPU kernels.")
    
    start_total = time.time()
    final_factors = None
    all_counts = {}
    
    for attempt in range(1, 21):
        print(f"      > Running Shot {attempt}...", end="", flush=True)
        t0 = time.time()
        
        # RUN 1 SHOT
        result = sim.run(qc_t, shots=1).result()
        
        dt = time.time() - t0
        counts = result.get_counts()
        bitstring = list(counts.keys())[0]
        
        # Analyze
        val = int(bitstring, 2)
        r = get_order_from_meas(val, t_eff, a, N)
        valid = (pow(a, r, N) == 1)
        
        status = "Fail"
        if valid:
            factors = get_factors_from_r(a, r, N)
            if factors:
                status = f"SUCCESS -> {factors}"
                final_factors = factors
            elif r % 2 != 0: status = "Odd Period"
            elif pow(a, r//2, N) == N-1: status = "Trivial"
        
        print(f" Done ({dt:.2f}s) | r={r:<4} | {status}")
        
        all_counts[bitstring] = all_counts.get(bitstring, 0) + 1
        if final_factors: break

    total_time = time.time() - start_total
    return all_counts, final_factors, total_time

def plot_counts(counts, title, filename):
    if not counts: return
    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values(), color="#003366", edgecolor="black")
    plt.xlabel("Measured Bitstring")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"[Plot] Saved to {os.path.abspath(filename)}")

if __name__ == "__main__":
    print("=======================================================")
    print("   SHOR'S ALGORITHM:          ")
    print("=======================================================")
    
    while True:
        try:
            val = input("\nEnter N (or 0 to quit): ").strip()
            if not val: continue
            N = int(val)
            if N == 0: break
            
            a = int(input(f"Enter a (coprime to {N}): ").strip())
            
            print("-" * 50)
            counts, factors, dt = run_simulation_streaming(N, a)
            print("-" * 50)
            
            print(f"Total Time: {dt:.4f}s")
            if factors:
                print(f"*** FINAL RESULT: {factors} ***")
            else:
                print("*** FAILED. ***")
            
            if counts:
                 plot_counts(counts, f"Shor N={N}, a={a}", "shor_stream_final_plot.png")
            
        except ValueError: print("Invalid input")
        except Exception as e: print(f"[Error] {e}")