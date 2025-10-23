# === PART 4 (Vasu) --- Analyze Resource Usage ===

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
