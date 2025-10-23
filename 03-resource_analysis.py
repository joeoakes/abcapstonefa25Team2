# === PART 4 (Vasu) --- Analyze Resource Usage ===
# Collect Data
# This cell runs existing Shor code many times and saves results to results.csv.

import csv, time
from qiskit_aer import Aer
from qiskit import transpile

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

# === PART 5 (Thomas + Jon) --- Data Visualization ===
#THIS WONT WORK IF YOU JUST PUT IT ALONE BUT I WILL PLACE IT HERE 
# === (Thomas) Graphs Comparing Noise (Baseline)===
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




# === (Thomas) Graphs Comparing Noise (Noisy Run) ===
import matplotlib.pyplot as plt
import io, contextlib

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

        plt.show()  #display chart
