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

# === Noise and Coprime Setup ===
def find_best_coprime(N):
    candidates = [a for a in range(2, N) if gcd(a, N) == 1]
    random.shuffle(candidates)
    return candidates[0] if candidates else None

def get_noise_model(use_depol=True, use_amp=True, use_phase=True):
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

# === Build Iterative Semiclassical Circuit ===
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

def run_iterative_shor():
    print("=== Iterative Shor (Semiclassical QFT, 1-qubit) ===")
    N = int(input("Enter composite N to factor: "))
    a = find_best_coprime(N)
    print(f"Using coprime a = {a}")

    print("Select noise components (y/n):")
    use_depol = input("  Depolarizing noise? ").lower() == 'y'
    use_amp = input("  Amplitude damping?  ").lower() == 'y'
    use_phase = input("  Phase damping?      ").lower() == 'y'

    max_attempts = int(input("Max attempts until success: "))
    start_time = time.time()

    with open("attempt_log.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Attempt", "Bitstring", "Circuit Depth", "Gate Count", "Success"])

        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Attempt {attempt} ---")
            qc, t = shor_iterative_circuit(a, N)
            noise_model = get_noise_model(use_depol, use_amp, use_phase)
            sim = AerSimulator(noise_model=noise_model)

            tqc = transpile(qc, sim)
            depth = tqc.depth()
            gates = dict(tqc.count_ops())
            print(f"[Info] Circuit depth: {depth}")
            print(f"[Info] Total gates: {gates}")

            result = sim.run(tqc, shots=1).result()
            counts = result.get_counts()
            print("counts =", counts)

            for b in counts:
                print("result bits:", b)
                factors = estimate_order(b, a, N)
                success = factors is not None
                writer.writerow([attempt, b, depth, gates, success])

                if success:
                    elapsed = time.time() - start_time
                    print(f"[SUCCESS] Found factors: p={factors[0]}, q={factors[1]}, p*q={factors[0] * factors[1]}")
                    print(f"[Time] Success after {elapsed:.2f} seconds.")
                    tqc.draw(output='mpl', filename='transpiled_circuit_success.png')
                    plt.close()
                    qc.draw(output='mpl', filename='raw_circuit_success.png')
                    plt.close()
                    return

        print("\n[FAIL] All attempts completed with no valid factors found.")

if __name__ == "__main__":
    run_iterative_shor()