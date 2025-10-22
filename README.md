# Breaking Crypto with Quantum Simulation - Team 2

This project, developed by Team 2 (Team Yoda) explores how quantum computing can challenege classical encryption by simulationg **Shor's Algorithm** - a quantum method for factoring the large intergers commonly using in RSA encryption. Using both Python and Qiskit, our team has developed a simulator that **performs quantum factoring** on smaller RSA problems, measures the **error and noise effects** and success probability with each run, and has been **optimized** after each run.

## Project Structure (WIP)

/abcapstonefa25Team2
│
├── shor_noiseless_demo.py # Serves as the control for the simulation
├── /data # Stores output metrics and figures
│ ├── results.csv
│ └── plots/
└── README.md # Project documentation

## Setup

### Dependencies

* Python 3.9 or higher  
* Required libraries:
```
pip install qiskit matplotlib numpy
```

### Executing program (WIP)

* Clone or download this repository  
* Navigate to the project folder  
* Run the demo programs using the following commands
```
git clone ---
cd abcapstonefa25Team2
---
```

## Help

* If Qiskit backend errors occur:
```
pip install --upgrade qiskit
```
* If simulation is slow:
  * Reduce the number of qubits or test a smaller integer (e.g., N = 15)  
  * Disable noise models or reduce shots (`shots=512`)  
  * Run only the core factoring circuit for validation

## Version History

* 0.1
    * Initial Release
 
## Professor

* Joa Oakes - CMPSC 488/IST 440W FW25

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


