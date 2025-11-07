# Breaking Crypto with Quantum Simulation - Team 2

## Project Overview

This project, developed by Team 2 (Team Yoda) explores how quantum computing can challenege classical encryption by simulationg **Shor's Algorithm** - a quantum method for factoring the large intergers commonly using in RSA encryption. 

Using both Python and Qiskit, our team has developed a simulator that **performs quantum factoring** on smaller RSA problems, measures the **error and noise effects** and success probability with each run, and has been **optimized** after each run.

## Project Objectives

* **Demonstrate Quantum Factoring:**
  Build and simulate a wroking version of Shpr's Algortihm in PYhton using Qiskit.
* **Model Realistic Quantum Behavior:**
  Introduce noise models to understand how errors and decoherence affect success rates.
* **Optimize Quantum Circuits:**
  Reduce number of gates and circuit depth through different forms of optimization.
* **Estimate Resource Costs:**
  Approximate the number of qubits, gate operations, runtime, and energy consumption necessary for large-scale quantum factoring.

## Project Structure

```
/abcapstonefa25Team2
│
├── 01-shor_noiseless_demo.py # Serves as the control for the simulation
├── 02-shor_with_noise_demo.py # Simulates with noise functions added
├── 03-resource_analysis.py # Estimates qubits, gates, runtime, and energy
├── /data # Stores output metrics and figures
│ ├── results.csv
│ └── plots/
└── README.md # Project documentation
```

## Setup

### Technologies Used

* Python 3.9 or higher  
* Required libraries:
```
pip install qiskit matplotlib numpy pylatexenc
```

### How to Run

* Clone or download this repository  
* Navigate to the project folder  
* Run the demo programs using the following commands
```
git clone https://github.com/joeoakes/abcapstonefa25Team2
cd abcapstonefa25Team2
python 01-shor_noiseless_demo.py
```

### How to Test

* If Qiskit backend errors occur:
```
pip install --upgrade qiskit
```
* If simulation is slow:
  * Reduce the number of qubits or test a smaller integer (e.g., N = 15)  
  * Disable noise models or reduce shots (`shots=512`)  
  * Run only the core factoring circuit for validation
 
## Results and Screen Captures

<img width="390" height="256" alt="Noiseless Graph" src="https://github.com/user-attachments/assets/dbbed5fc-11b1-456f-9af1-636005837a8e" />


<img width="390" height="256" alt="Noise Graph" src="https://github.com/user-attachments/assets/048460e9-89bb-44e4-a143-eece043fb8ff" />

## Ethics

### Background

A majority of our life that connects to the internet, runs off a sort of encryption, whether that be your credit cards, your password to banking systems, purchasing something off of amazon, and much more. But a majority of our online traffic, through HTTPS is secured using a type of encryption called RSA.

Not just HTTPS, but RSA secures the connection for emails, SSH, and even VPN’s to name a few. Roughly 70% of websites across the internet still use RSA to encrypt traffic between servers and users. A common example is the connection used when using Google which is secured by RSA, as shown below.

<img width="513" height="370" alt="SCR-20251106-gak" src="https://github.com/user-attachments/assets/d5b2ec57-d542-4ca7-a566-1c9b4067b004" />

RSA is done in 3 steps:
* Key Generation: Choose two large primes (p, q), compute n = p × q, find public exponent e and private exponent d.
* Encryption / Signing: Convert the message to a number m and compute the ciphertext (c) as, c = mᵉ mod n.
* Decryption / Verification: compute m = cᵈ mod n (or verify by checking sᵉ mod n = m).

The public key is what’s important in this, that key, keeps the prime numbers unknown and hard to factor when in a large numbers, which is what can bring you the decryption key. Classical computers would take tens of billions of years to break RSA-2048, the sun would explode before we can try to crack a key that large. But with quantum computers that process to factor the public key in theory would only take hours to possibly days

### Is this Ethical?

In short, no. The second we get the ability to crack RSA with quantum computers, we risk the infrastructure, integrity and safety of almost the entire internet as a whole, besides the political and human impact it would cause. There is no estimate on how much money the world would lose and how much would be impacted, but when processed it could be in the billions into trillions. All sorts of data can be exposed and breached when tapped into a connection between the server and the client. Think credit cards details once secure behind a payment screen encrypted by RSA now exposed and decrypted, sold to the highest bidder, and the emails that can expose national security risks. 


## Version History

* 0.1
    * Initial Release
 
## Course Information

* Course: CMPSC 488/IST 440W
* Institution: Penn State Abington
* Term: Fall/Winter 2025
* Professor: Joe Oakes

## Team 2 Members

* **Jonathan Cunningham** - Project Lead & Coordinator  
* **Haroun Ramadan** - Ethics Verification & Testing Lead  
* **Marco Isabella** - Circuit Optimization Engineer  
* **Marco Ramirez** - Noise & Error Modeling  
* **Tao Geng** - Quantum Algorithm Specialist  
* **Thomas McConnell** - Data Visualization & Report Designer  
* **Vasu Patel** - Resource Estimation Analyst  

## References

* Qiskit Documentation – [https://docs.quantum.ibm.com](https://docs.quantum.ibm.com)

## License Details
```
* Insert Here *
```

