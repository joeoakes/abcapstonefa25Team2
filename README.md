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

<img width="390" height="256" alt="IMG_6321" src="https://github.com/user-attachments/assets/dbbed5fc-11b1-456f-9af1-636005837a8e" />


<img width="586" height="384" alt="IMG_5566" src="https://github.com/user-attachments/assets/048460e9-89bb-44e4-a143-eece043fb8ff" />


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

