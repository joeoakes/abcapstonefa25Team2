import numpy as np
from math import gcd, log2, ceil, pi
from fractions import Fraction
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random
import csv
import time
import os

# === Noise and Coprime Setup ===
def find_best_coprime(N):
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    random.shuffle(candidates)
    return candidates[0] if candidates else None

def get_noise_model(use_depol, use_amp, use_phase):
    noise_model = NoiseModel()
    if use_depol:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["u3", "rx", "ry", "rz"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    if use_amp or use_phase:
        amp = amplitude_damping_error(0.01) if use_amp else None
        phase = phase_damping_error(0.01) if use_phase else None
        if amp and phase:
            noise_model.add_all_qubit_quantum_error(amp.compose(phase), ["u3", "rx", "ry", "rz"])
        elif amp:
            noise_model.add_all_qubit_quantum_error(amp, ["u3", "rx", "ry", "rz"])
        elif phase:
            noise_model.add_all_qubit_quantum_error(phase, ["u3", "rx", "ry", "rz"])
    return noise_model

def shor_iterative_circuit(a, N):
    n = ceil(log2(N))
    t = 2 * n
    c = ClassicalRegister(t)
    work = QuantumRegister(n, name="work")
    control = QuantumRegister(1, name="control")
    qc = QuantumCircuit(control, work, c)
    qc.x(work[n - 1])

    for k in reversed(range(t)):
        qc.h(control[0])
        exp = pow(a, 2 ** k, N)
        qc.barrier()
        qc.measure(control[0], c[k])
        qc.reset(control[0])
    return qc, t

def estimate_order(bitstring, a, N):
    y = int(bitstring, 2)
    phase = y / (2 ** len(bitstring))
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    if r % 2 != 0 or pow(a, r, N) != 1:
        return None
    guess1 = gcd(pow(a, r // 2) - 1, N)
    guess2 = gcd(pow(a, r // 2) + 1, N)
    if 1 < guess1 < N and 1 < guess2 < N and guess1 * guess2 == N:
        return (guess1, guess2)
    return None

def confirm_gpu_enabled():
    try:
        test_sim = AerSimulator(method='statevector', device='GPU')
        if 'gpu' in test_sim.name.lower():
            print("[✔] GPU simulator confirmed:", test_sim.name)
            return True
        else:
            print("[✘] GPU fallback detected.")
            return False
    except Exception as e:
        print("[✘] Error checking GPU backend:", e)
        return False

def run_all_noise_modes():
    print("=== Run All Noise Combinations Until Success (GPU/CPU Comparison) ===")
    N = int(input("Enter composite N to factor: "))
    a = find_best_coprime(N)
    print(f"Using coprime a = {a}")

    confirm_gpu_enabled()

    backends = [("GPU", 'GPU'), ("CPU", 'CPU')]

    noise_modes = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]

    combined_summary = []

    for backend_name, device_type in backends:
        print(f"\n### Running on {backend_name} backend ###")
        summary_data = []

        for d, a_d, p in noise_modes:
            label = "NONE"
            if d or a_d or p:
                label = "".join([
                    "DEP" if d else "",
                    "_AMP" if a_d else "",
                    "_PHASE" if p else ""
                ])
            label = f"{label}_{backend_name}"
            print(f"\n--- Noise Mode: {label} ---")
            start_time = time.time()

            with open(f"attempt_log_{label}.csv", mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Attempt", "Bitstring", "Circuit Depth", "Gate Count", "Success", "Time (s)", "Noise Label", "Backend"])
                attempt = 1
                while True:
                    qc, t = shor_iterative_circuit(a, N)
                    noise_model = get_noise_model(d, a_d, p)
                    sim = AerSimulator(method='automatic', device=device_type, noise_model=noise_model)
                    tqc = transpile(qc, sim)
                    depth = tqc.depth()
                    gates = dict(tqc.count_ops())
                    result = sim.run(tqc, shots=1).result()
                    counts = result.get_counts()
                    b = list(counts.keys())[0]
                    factors = estimate_order(b, a, N)
                    success = factors is not None
                    elapsed = time.time() - start_time

                    writer.writerow([attempt, b, depth, gates, success, f"{elapsed:.2f}", label, backend_name])

                    if success:
                        print(f"[SUCCESS {label}] Found: {factors}, time={elapsed:.2f}s, attempts={attempt}")
                        tqc.draw(output='mpl', filename=f'transpiled_success_{label}.png')
                        plt.close()
                        qc.draw(output='mpl', filename=f'raw_success_{label}.png')
                        plt.close()
                        summary_data.append((label, elapsed, sum(gates.values()), backend_name))
                        break
                    attempt += 1

        combined_summary.extend(summary_data)

    # Split into GPU and CPU plots
    gpu_data = [entry for entry in combined_summary if entry[3] == "GPU"]
    cpu_data = [entry for entry in combined_summary if entry[3] == "CPU"]

    def make_bar_plot(data, title, filename, ylabel):
        labels, vals = zip(*[(x[0], x[1 if ylabel == "Seconds" else 2]) for x in data])
        fig, ax = plt.subplots()
        ax.bar(labels, vals)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)

    make_bar_plot(gpu_data, "GPU: Time to Success by Noise Mode", "gpu_time.png", "Seconds")
    make_bar_plot(cpu_data, "CPU: Time to Success by Noise Mode", "cpu_time.png", "Seconds")
    make_bar_plot(gpu_data, "GPU: Gate Count by Noise Mode", "gpu_gates.png", "Gate Count")
    make_bar_plot(cpu_data, "CPU: Gate Count by Noise Mode", "cpu_gates.png", "Gate Count")

    print("\n✅ Completed CPU/GPU noise comparison. Plots and CSV logs generated.")

if __name__ == "__main__":
    run_all_noise_modes()
