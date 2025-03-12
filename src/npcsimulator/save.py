import numpy as np
import h5py

def save_data(filename, true_emitters, unlabelled_emitters, observed_measurements, noise_emitters, edges):
    """
    Saves emitter and intensity data to CSV or HDF5 based on the file extension in `filename`,
    and includes emitter adjacency edges.

    :param filename: Name of the output file (e.g., 'data.csv' or 'data.h5').
    :param true_emitters: Numpy array of the labelled emitters.
    :param unlabelled_emitters: Numpy array of unlabelled emitters.
    :param observed_measurements: Numpy array of observed measurements.
    :param noise_emitters: Numpy array of noise emitters with coordinates.
    :param edges: List of tuples representing adjacency edges (connections between emitters).
    """
    if filename.endswith('.csv'):
        # Prepare true emitters data
        true_data = np.column_stack(
            (true_emitters, np.full(true_emitters.shape[0], 'labelled')))
        # Prepare unlabelled emitters data
        unlabelled_data = np.column_stack(
            (unlabelled_emitters, np.full(unlabelled_emitters.shape[0], 'unlabelled')))
        # Combine for ground truth
        ground_truth = np.vstack((true_data, unlabelled_data))

        # Prepare observed measurement data
        observed_data = np.column_stack(
            (observed_measurements, np.full(observed_measurements.shape[0], 'observed')))
        # Combine into measurement data
        measurement_data = np.vstack(observed_data)

        # Prepare noise emitters data
        noise_data = np.column_stack(
            (noise_emitters, np.full(noise_emitters.shape[0], 'clutter')))

        # Combine all data into a single array
        all_data = np.vstack((ground_truth, measurement_data, noise_data))

        # Create header for the CSV file
        header = 'x,y,z,type\n'

        # Save emitter data to CSV
        np.savetxt(filename, all_data, delimiter=',', header=header, comments='', fmt='%s')
        print(f"Emitter data saved to {filename}")

        # Save edges separately as CSV
        edge_filename = filename.replace('.csv', '_edges.csv')
        np.savetxt(edge_filename, edges, delimiter=',', header='node1,node2', comments='', fmt='%d')
        print(f"Edges saved to {edge_filename}")

    elif filename.endswith('.h5'):
        # Save to HDF5
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('true_emitters', data=true_emitters)
            hf.create_dataset('unlabelled_emitters', data=unlabelled_emitters)
            hf.create_dataset('observed_measurements', data=observed_measurements)
            hf.create_dataset('clutter', data=noise_emitters)
            hf.create_dataset('edges', data=np.array(edges, dtype=np.int32))  # Save edges as an integer dataset
        print(f"Data and edges saved to {filename}")

    else:
        print("Unsupported file format. Please use .csv or .h5")
