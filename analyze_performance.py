import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Branch diagram plotter {{{
def plot_branch_diagram(dataframes, labels, output_file, plot_type="redshift"):
    """
    Plot a branching diagram for multiple dataframes.

    Parameters:
    - dataframes: List of pandas DataFrames.
    - labels: List of labels for each DataFrame.
    - plot_type: "redshift" or "scale_factor"
    """
    plt.figure(figsize=(12, 7))

    # Plot the main branch (first dataframe)
    main_data = dataframes[0]
    if plot_type == "scale_factor":
        main_data['value'] = 1 / (1 + main_data['redshift'])
        ylabel = 'Scale Factor (a)'
    else:
        main_data['value'] = main_data['redshift']
        ylabel = 'Redshift (z)'

    plt.plot(main_data['time'] - main_data['time'].iloc[0], main_data['value'], label=labels[0])

    # Adjust and plot subsequent branches
    for df, label in zip(dataframes[1:], labels[1:]):
        adjusted_df = adjust_time_for_branching_and_trim(main_data, df, plot_type)
        if plot_type == "scale_factor":
            adjusted_df['value'] = 1 / (1 + adjusted_df['redshift'])
        else:
            adjusted_df['value'] = adjusted_df['redshift']
        plt.plot(adjusted_df['adjusted_time'] - main_data['time'].iloc[0], adjusted_df['value'], label=label)

    plt.xlabel('Adjusted Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs. Adjusted Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    # plt.show()
#}}}

# Convert labels to tables, usable in the functions above {{{
def labels_to_tables(labels):
    tables = []
    for label in labels:
        filename = '../output/'+label+'/performance_report.csv'
        table = pd.read_csv(filename)
        # Delete rows where the second column value is negative
        table = table[table.iloc[:, 1] >= 0]
        tables.append(table)
    return tables
#}}}

label1 = '2023.09.12:1'
##label2 = '2023.09.11:2'
##label3 = '2023.09.11:1'
#label2 = '2023.09.12:13'
label2 = '2023.09.13:7'
label3 = '2023.09.13:9'
#label3 = '2023.09.12:3'
#label4 = '2023.09.12:4'
#label5 = '2023.09.12:5'
#label1 = '2023.09.12:6'
#label2 = '2023.09.12:7'
#label3 = '2023.09.12:8'
#label4 = '2023.09.12:9'
#label5 = '2023.09.12:10'
#label6 = '2023.09.12:11'
#label7 = '2023.09.12:12'
#labels = [label1, label2, label3, label4, label5]
#labels=[label1, label2, label3, label4, label5, label6, label7]
labels=[label1, label2, label3]

output_file = '/cita/d/www/home/vpustovoit/plots/performance_analyzis.png'

tables = labels_to_tables(labels)
#real_abels = ['OpenMP=2', label2, label3, label4, label5]
real_labels = ['No Metals', 'With Metals, 198 MPI, OpenMP=2', 'With Metals, 96 MPI, OpenMP=4']
#real_labels = [ 'OpenMP=2, 128 mpi processes', 
#                'OpenMP=4, 64 mpi processes', 
#                'OpenMP=4, 128 mpi processes', 
#                'OpenMP=16, 128 mpi processes', 
#                'OpenMP=16, 256 MPI processes',
#                'OpenMP=1, 256 MPI processes',
#                'OpenMP=8, 128 MPI processes']

plot_branch_diagram(tables, real_labels, output_file, plot_type="scale_factor")
