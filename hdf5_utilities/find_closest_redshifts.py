import csv
import sys

def find_closest_snapshot(file_path, target_redshift):
    """
    Finds the snapshot with the closest redshift to the target.

    Parameters:
    - file_path: The path to the CSV file containing snapshot info.
    - target_redshift: The redshift to find the closest match for.

    Returns:
    - The name of the snapshot with the closest redshift.
    """
    # Initialize variables to store the best match
    closest_snapshot = None
    smallest_diff = float('inf')
    
    # Open and read the CSV file
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            snapshot_redshift = float(row['Redshift'])
            diff = abs(snapshot_redshift - target_redshift)
            
            # Update the best match if this snapshot is closer
            if diff < smallest_diff:
                smallest_diff = diff
                closest_snapshot = row['Snapshotfile']
    
    return closest_snapshot

if __name__ == "__main__":
    
    # Check if at least one argument is provided (first argument is the script name)
    if len(sys.argv) > 2:
        csv_file_location = sys.argv[1]
        target_redshift = float(sys.argv[2])
    else:
        print("Need 2 arguments: location of the csv file and target redshfit")
        sys.exit(1)  # Exit the script with an error code

    closest_redshift = find_closest_snapshot(csv_file_location, target_redshift) 
    print(closest_redshift)

