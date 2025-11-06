''' # required dependancies
!pip install qiskit --quiet
!pip install qiskit_aer --quiet
!pip install qiskit_ibm_runtime --quiet
'''
# === PART 3 (Marcos) --- Introducing Noise ===
import math
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, errors
import os
import matplotlib.pyplot as plt
import io, contextlib


def cswap_decomp(qc, c, a, b):
    qc.ccx(b, c, a)
    qc.ccx(a, c, b)
    qc.ccx(b, c, a)

def iqft_in_place(qc, qubits):
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n - 1, -1, -1):
        for k in range(j + 1, n):
            qc.cp(-math.pi / (2 ** (k - j)), qubits[k], qubits[j])
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
            qc.cx(ctrl, work[0])
            qc.cx(ctrl, work[1])
            qc.cx(ctrl, work[2])
            qc.cx(ctrl, work[3])

def controlled_U(qc, controls, work, a):
    for k, ctrl in enumerate(controls):
        apply_controlled_mul_a_mod_15(qc, ctrl, work, a, 2 ** k)

def continued_fraction_phase_estimate(meas_value, t):
    phase = meas_value / (2 ** t)
    return Fraction(phase).limit_denominator(2 ** t)

def try_order_from_measure(meas, t, a, N):
    frac = continued_fraction_phase_estimate(meas, t)
    if frac.denominator == 0:
        return None
    r = frac.denominator
    for candidate in [r, 2*r, 3*r, 4*r]:
        if pow(a, candidate, N) == 1:
            return candidate
    return None

def try_factors_from_r(a, r, N):
    if r is None or r % 2 == 1:
        return None
    x = pow(a, r // 2, N)
    if x in [1, N - 1, 0]:
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
    qc.x(counting)  # initialize |1> in work register
    for i in range(counting):
        qc.h(i)
    controlled_U(qc, range(counting), list(range(counting, counting + work)), a)
    iqft_in_place(qc, list(range(counting)))
    qc.measure(range(counting), range(counting))
    return qc

# https://quantum.cloud.ibm.com/docs/en/guides/build-noise-models
def create_simple_noise_model():

    noise_model = NoiseModel()
    # 1-qubit depolarizing error (~0.1%)
    depol_1 = errors.depolarizing_error(0.001, 1)

    # 2-qubit depolarizing error (~1%)
    depol_2 = errors.depolarizing_error(0.01, 2)

    # 3-qubit depolarizing error (~1.5%)
    depol_3 = errors.depolarizing_error(0.015, 3)

    # Readout error (~10%)
    readout_err = errors.ReadoutError([[0.9, 0.1], [0.1, 0.9]])

    # Assign to gates according to number of qubits
    noise_model.add_all_qubit_quantum_error(depol_1, ['h', 'x'])
    noise_model.add_all_qubit_quantum_error(depol_2, ['cx', 'cp'])
    noise_model.add_all_qubit_quantum_error(depol_3, ['ccx'])

    # Add measurement noise
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model



def shor_with_noise(a, N=15, t=8, shots=512, noisy=False):
    if math.gcd(a, N) != 1:
        print(f"gcd({a},{N}) != 1:", math.gcd(a, N))
        return

    qc = build_shor_circuit(a, N, t)

    if noisy:
        noise_model = create_simple_noise_model()
        sim = Aer.get_backend("aer_simulator_statevector")
        qc_t = transpile(qc, sim)
        print("\n Running with noise model...")
        result = sim.run(qc_t, shots=shots, noise_model=noise_model).result()
    else:
        sim = Aer.get_backend("qasm_simulator")
        qc_t = transpile(qc, sim)
        print("\n Running ideal (noiseless) simulation...")
        result = sim.run(qc_t, shots=shots).result()

    counts = result.get_counts()
    for bitstring, count in counts.items():
        meas_val = int(bitstring[::-1], 2)  # reverse bit order
        r = try_order_from_measure(meas_val, t, a, N)
        if r is None:
            continue
        factors = try_factors_from_r(a, r, N)
        if factors:
            p, q = factors
            print(f"SUCCESS: {p} × {q} = {N}  (from {bitstring}, {count} shots)")
            return counts
    print("No non-trivial factors found.")
    return counts


if __name__ == "__main__":
    for a in [2, 7]:
        print("\n" + "=" * 60)
        print(f"Running Shor for N=15, a={a}")
        print("\nIdeal simulation:")
        shor_with_noise(a, N=15, t=8, shots=512, noisy=False)
        print("\nNoisy simulation:")
        shor_with_noise(a, N=15, t=8, shots=512, noisy=True)



# === (Thomas) Graphs Comparing Noise (Noisy Run) ===
if __name__ == "__main__":  #only runs if above code executes correctly
    a = 7  #the base value 'a' used in Shor's algorithm
    buf = io.StringIO()

    # Hide printed output like “Candidate period r = …”
    with contextlib.redirect_stdout(buf):
        counts = shor_with_noise(a=a, N=15, t=8, shots=1024, noisy=True)  #run the noisy simulation only

    if counts:  #only plot if valid data
        total_shots = sum(counts.values())  #compute total number of measurements
        probabilities = {state: count / total_shots for state, count in counts.items()}  #convert counts to probabilities

        N_show = 10  # choose how many of the most frequent results to show (usually 4–6 but 10 for safety)
        #sort bitstrings by probability (highest first) and keep top N
        top_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:N_show])

        plt.figure(figsize=(6, 4))  #create new window and set size
        bars = plt.bar(top_probs.keys(), top_probs.values(), color='red')  #draw bar chart for noisy results
        plt.title(f"Top Measurement Outcomes for a={a} (With Noise)")  #title showing noisy condition
        plt.xlabel("Measured Bitstring")  #label horizontal axis
        plt.ylabel("Probability")  #label vertical axis
        plt.xticks(rotation=45)  #rotate x-axis labels for readability
        plt.tight_layout()  #adjust layout so nothing overlaps

        #loop through each bar and print its probability percentage above it
        for bar, val in zip(bars, top_probs.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,  #horizontally center text above bar
                bar.get_height(),  #position text
                f"{val * 100:.1f}%",  #show value as percentage
                ha='center',
                va='bottom',
                fontsize=10,
            )
    plt.savefig("noisy_plot.png")
    print("saved noisy_plot.png in", os.getcwd())
       # plt.show()  #display chart
