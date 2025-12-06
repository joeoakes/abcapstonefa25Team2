# Circuit Visualization Tool

import os
from math import log2, ceil
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# modular exponentiation (visual structure only)
def modexp_block(qc, control, target, exponent):
    qc.crz(2 ** (-exponent), control, target)

# Iterative Shor skeleton circuit
def shor_circuit(a, N):
    n = ceil(log2(N))
    t = 2 * N

    # registers
    c = ClassicalRegister(t)
    work = QuantumRegister(n, name="work")
    control = QuantumRegister(1, name="control")

    qc = QuantumCircuit(control, work, c)

    # initialize work register
    qc.x(work[n - 1])

    # iterative phase estimation loop
    for k in reversed(range(t)):
        qc.h(control[0])

        # toy controlled-U operation
        modexp_block(qc, control[0], work[0], k)

        qc.measure(control[0], c[k])
        qc.reset(control[0])

    return qc, t

def main():
    # output folder
    out_dir = "circuit_images"

    # small safe values for visualization
    values = [
        (15, 2),
        (21, 4),
        (33, 5),
    ]

    for N, a in values:
        qc, _ = shor_circuit(a, N)
        filename = os.path.join(out_dir, f"shor_circuit_N{N}_a{a}.png")
        print(f"Generating {filename}...")
        qc.draw("mpl", filename=filename)

    print("PNG files in directory: (circuit_images)")

if __name__ == "__main__":
    main()