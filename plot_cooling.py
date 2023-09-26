import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('Agg')

def plot_csv_data(input_filepath, label, color):

    # Read the CSV file into a DataFrame, considering the header
    df = pd.read_csv(input_filepath, dtype={'logT': float, 'Lambda': float, 'logLambda': float})
    print(f"Data from {input_filepath}:\n{df.head()}")
    if 'Lambda' in df.columns:
        Lambda = df['Lambda']
        Lambda = abs(Lambda)
    elif 'logLambda' in df.columns:
        Lambda = 10**df['logLambda']  
    else:
        print("Neither 'Lambda' nor 'logLambda' columns found in DataFrame")
    logLambda = np.log(Lambda) / np.log(10)

    #print(f"Plotting data from {input_filepath}:\n{logLambda}")
    # Check if data falls within the plotting range
    if df['logT'].min() > 8.5 or df['logT'].max() < 4.1:
        print(f"Data from {input_filepath} is out of the x-axis range.")
        return  # Skip this file

    if logLambda.min() > -20 or logLambda.max() < -24:
        print(f"Data from {input_filepath} is out of the y-axis range.")
        return  # Skip this file
    
    # Create the plot
    plt.plot(df['logT'], logLambda, c=color, label=label)



# This is how you would call the function with an input and output filepath
def plot_cooling(rho_id):
    xmin, xmax = (3, 10)
    #xmin, xmax = (4.1, 8.5)
    plt.figure(figsize=(8, 8))
    plot_title='Lambda vs logT, log(rho) = ' + str(rho_id)
    plt.title(plot_title)
    plt.xlabel('logT')
    plt.ylabel('Lambda')
    # Adjust axis ticks for less crowding
    x_ticks = np.linspace(xmin, xmax, 11)
    plt.xticks(x_ticks, rotation=45)
    plt.xlim(xmin, xmax)
    plt.ylim(-24, -20)
    #plt.yticks(y_ticks, rotation=45)
    
    
    output_filepath = "/cita/d/www/home/vpustovoit/plots/cooling/cooling" + str(rho_id) + ".png"
    input_filepath = "../gizmo/cooling_rho_" + str(rho_id) + ".csv"
    plot_csv_data("../gizmo/cooling.csv", "nil", "blue")
    plot_csv_data(input_filepath, "0","red")
    #plot_csv_data("../gizmo/cooling_molec.csv", "molecular","orange")
    plot_csv_data("../gizmo/cooling_book.csv", "ref, nil","black")
    plot_csv_data("../gizmo/cooling_book_0.csv", "ref, 0","green")
    #plot_csv_data("../gizmo/cooling_AG.csv", "AG","purple")
    
    # Save the plot
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

rho_array = np.linspace(-30, -10, 21)
print(rho_array)
for i in range(0, np.size(rho_array)):
    plot_cooling(int(rho_array[i]))
