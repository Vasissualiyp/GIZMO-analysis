import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # or try 'TkAgg' if you want to display plots interactively
import matplotlib.pyplot as plt

# Set which runs you are interested in and which job_path you want to set for them in the final plot.
# The path to runs can be changed in job_path_to_tables function
labels_full = [
               #[ '2023.10.25:7', 'New compilation parameters, O2 (wrong libraries)'],
               #[ '2023.11.23:1', '2^0'],
               #[ '2023.11.23:2', '2^1'],
               #[ '2023.11.21:6', '2^2'],
               #[ '2023.11.21:7', '2^3'],
               #[ '2023.11.21:8', '2^4'],
               #[ '2023.11.21:9', '2^5'],
               [ '2023.11.21:14', '1', 1],
               [ '2023.11.21:15', '2', 2],
               [ '2023.11.21:16', '3', 3],
               [ '2023.11.23:7',  '4', 4],
               #[ '2023.11.21:12', '2^6'],
               #[ '2023.11.21:13', '2^7'],
               #[ '2023.11.21:14', '2^8'],
               #[ '2023.10.25:10', 'New compilation parameters, O3'],
               #[ '2023.10.25:9', 'Old compilation parameters, O2'],
               #[ '2023.10.25:6', 'Old compilation parameters, O3']
               ]

# Location of the output file
output_file = './plots/scaling.png'
output_file_strongscale = '/cita/d/www/home/vpustovoit/plots/strong_scaling.png'
plt_title = 'Strong scaling test, starq'

cores_per_node = 128 # 40

#--------------------SOURCE CODE------------------- {{{
# Find branching points {{{
def find_branching_point(main_data, branch_data, plot_type="redshift"):
    """
    Find the branching point of the branch_data from the main_data.
    Returns the time in main_data where the redshift/scale factor is closest to the starting redshift/scale factor of branch_data.
    """
    if plot_type == "scale_factor":
        main_data['value'] = 1 / (1 + main_data['redshift'])
        branch_data['value'] = 1 / (1 + branch_data['redshift'])
        start_value = branch_data['value'].iloc[0]
    else:
        main_data['value'] = main_data['redshift']
        branch_data['value'] = branch_data['redshift']
        start_value = branch_data['redshift'].iloc[0]
    
    closest_idx = np.argmin(np.abs(main_data['value'] - start_value))
    return main_data['time'].iloc[closest_idx]
#}}}

# Time adjustment for branches {{{
def adjust_time_for_branching_and_trim(main_data, branch_data, plot_type="redshift"):
    """
    Adjust the time of branch_data based on its branching point from main_data, trim the data if needed, and set the starting time
    to match the branching point.
    Returns the branch_data with adjusted time, trimmed values, and synchronized starting time.
    """
    branching_time = find_branching_point(main_data, branch_data, plot_type)
    
    # Trim the data if it starts at an earlier redshift/scale factor than the main data
    if plot_type == "scale_factor":
        main_start_value = 1 / (1 + main_data['redshift'].iloc[0])
        mask = branch_data['value'] >= main_start_value
    else:
        main_start_value = main_data['redshift'].iloc[0]
        mask = branch_data['redshift'] <= main_start_value

    trimmed_df = branch_data[mask].copy()  # Explicitly create a copy of the trimmed data
    
    # Adjust the time so that the starting time of the trimmed data matches the branching point from the main data
    time_offset = branching_time - trimmed_df['time'].iloc[0]
    trimmed_df['adjusted_time'] = trimmed_df['time'] + time_offset
    
    return trimmed_df
#}}}

# Truncate the dataframe to have max time = time of achieving a threshold value {{{
def truncate_dataframes_to_threshold_value(dataframes):

    threshold_value = get_threshold_for_scaling(dataframes)

    filtered_dfs = []
    for df in dataframes:
        truncated_df = df[df['redshift'] > threshold_value]
        filtered_dfs.append(truncated_df)

    return filtered_dfs
#}}}

# Find threshold value for comparison of scaling {{{
def get_threshold_for_scaling(dataframes):
    threshold = 0
    for df in dataframes:
        minred = min(df['redshift'])
        print(minred)
        if minred > threshold:
            threshold = minred
    print("Threshold is: " + str(threshold))
    return threshold
#}}}

# Branch diagram plotter {{{
def plot_branch_diagram(dataframes, labels, output_file, plot_type="redshift"):
    """
    Plot a branching diagram for multiple dataframes.

    Parameters:
    - dataframes: List of pandas DataFrames.
    - labels: List of job_path for each DataFrame.
    - plot_type: "redshift" or "scale_factor"
    """
    plt.figure(figsize=(12, 7))
    truncated_dfs = truncate_dataframes_to_threshold_value(dataframes)

    # Plot the main branch (first dataframe)
    main_data = truncated_dfs[0]
    if plot_type == "scale_factor":
        main_data['value'] = 1 / (1 + main_data['redshift'])
        ylabel = 'Scale Factor (a)'
    else:
        main_data['value'] = main_data['redshift']
        ylabel = 'Redshift (z)'

    plt.plot(main_data['time'] - main_data['time'].iloc[0], main_data['value'], label=labels[0])

    # Adjust and plot subsequent branches
    for df, label in zip(truncated_dfs[1:], labels[1:]):
        adjusted_df = adjust_time_for_branching_and_trim(main_data, df, plot_type)
        if plot_type == "scale_factor":
            adjusted_df['value'] = 1 / (1 + adjusted_df['redshift'])
        else:
            adjusted_df['value'] = adjusted_df['redshift']
        plt.plot(adjusted_df['adjusted_time'] - main_data['time'].iloc[0], adjusted_df['value'], label=label)

    plt.xlabel('Adjusted Time (s)')
    plt.ylabel(ylabel)
    #plt.ylim([0.005, 0.07]) 
    plt.title(ylabel + " vs. Adjusted Time, " + plt_title)

    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)

    return truncated_dfs
    # plt.show()
#}}}

# Get the times relevant for strong scaling from the dataframes {{{
def obtain_scaling(dataframes, labels, cores_per_node):
    times = []
    nodes = []
    for df, label in zip(dataframes, labels):
        maxt = max(df['time'])
        mint = min(df['time'])
        dtime = maxt - mint
        times.append(dtime)

        label_expression = label.replace('^', '**')
        nodes_current = eval(label_expression) # Get the number of nodes for the current df
        nodes.append(nodes_current)

        threshold = min(df['redshift'])

    cores = np.array(nodes) * cores_per_node
    
    return cores, times, threshold
#}}}
       
# Main funciton to plot strong scaling relation from the dataframes {{{
def plot_strong_scaling(dataframes, labels, output_file, cores_per_node, plot_type):
    secondary_cores = [128, 256, 384, 512, 640]
    secondary_scaling = [1., 1.73817239, 2.38241173, 2.65478842, 2.94401756]

    cores, times, threshold_red = obtain_scaling(dataframes, labels, cores_per_node)

    threshold_a = 1 / (1 + threshold_red)
    threshold_str = formatted_string = "{:.3f}".format(threshold_a)

    # Normalize the human time to the longest one
    longesttime = max(times) 
    print(f"The longest time is: {longesttime}")
    if plot_type == 'scale_factor':
        times_normalized = np.array(times) / longesttime
        scaling_factors = 1 / times_normalized
    elif plot_type == 'runtime':
        scaling_factors = times_normalized

    print("Cores:")
    print(cores)
    print("Times:")
    print(times)
    print("Scaling factors:")
    print(scaling_factors)

    # Create a log plot
    plt.figure(figsize=(10, 6))

    # primary machine
    plt.plot(cores, scaling_factors, marker='o', label='starq', linestyle='--')  

    # Set up secondary machine, if present
    if len(secondary_cores) < 2:
        # This part is needed for the plotting of ideal law
        secondary_cores = cores
        secondary_scaling = scaling

    # Find the boundaries of interest
    min_cores = min(min(cores), min(secondary_cores))
    max_cores = max(max(cores), max(secondary_cores))
    min_scaling = min(min(scaling_factors), min(secondary_scaling))

    # Plot secondary machine, if present
    if len(secondary_cores) > 1:

        if min_cores == min(cores):
            secondary_scaling = np.array(secondary_scaling) * min(secondary_cores) / min_cores
        else:
            scaling_factors = np.array(scaling_factors) * min(cores) / min_cores

        plt.plot(secondary_cores, secondary_scaling, marker='o', label='niagara', linestyle='--') 

    # Ideal law {{{
    # Create ideal law
    x_ideal = np.linspace(min_cores, max_cores, 1000)
    y_ideal = np.array(x_ideal) / min_cores

    # Plot 
    plt.plot(x_ideal, y_ideal, label = 'Ideal')  
    #}}}

    # Set the x-axis to be logarithmic
    plt.xscale('log')
    plt.yscale('log')

    # Adding labels and title
    plt.xlabel('Number of cores')

    if plot_type == 'scale_factor':
        plt.ylabel('Scaling factor, normalized to lowest ')
        plt.title('Speed up factor to get to a=' + threshold_str)
    elif plot_type == 'runtime':
        plt.ylabel('CPUhrs, normalized to lowest ')
        plt.title('Runtime to a=' + threshold_str)

    plt.legend()
    plt.savefig(output_file)
    
#}}}

# Convert job_path to tables, usable in the functions above {{{
def job_path_to_tables(labels):
    tables = []
    for label in labels:
        filename = '../output/'+label+'/performance_report.csv'
        table = pd.read_csv(filename)
        # Delete rows where the second column value is negative
        table = table[table.iloc[:, 1] >= 0]
        tables.append(table)
    return tables
#}}}
#--------------------MAIN------------------- }}}

job_path = [item[0] for item in labels_full]
plot_labels = [item[1] for item in labels_full]

tables = job_path_to_tables(job_path)

print(tables[0].columns)
truncated_dataframes = plot_branch_diagram(tables, plot_labels, output_file, plot_type="scale_factor")

plot_type = "scale_factor"
plot_strong_scaling(truncated_dataframes, plot_labels, output_file_strongscale, cores_per_node, plot_type)
