import numpy as np
import h5py

def save_data(filename, true_emitters, unlabelled_emitters, observed_measurements,
              unobserved_measurements,noise_emitters):
    """
    Saves emitter and intensity data to CSV or HDF5 based on the file extension in `filename`.

    :param filename: Name of the output file (e.g., 'data.csv' or 'data.h5').
    :param true_emitters: Numpy array of the true emitters (labelled).
    :param unlabelled_emitters: Numpy array of unlabelled emitters.
    :param observed_measurements: Numpy array of observed measurements.
    :param unobserved_measurements: Numpy array of unobserved measurements.
    :param noise_emitters: Numpy array of noise emitters with coordinates.
    """
    if filename.endswith('.csv'):
        # Prepare true emitters data
        true_data = np.column_stack(
            (true_emitters, np.full(true_emitters.shape[0],'labelled')))
        # Prepare unlabelled emitters data
        unlabelled_data = np.column_stack(
            (unlabelled_emitters, np.full(unlabelled_emitters.shape[0], 'unlabelled')))
        # Combine for ground truth
        ground_truth = np.vstack((true_data, unlabelled_data))

        # Prepare observed measurement data
        observed_data = np.column_stack((observed_measurements
                                           , np.full(observed_measurements.shape[0],
                                                     'observed')))
        # Prepare unobserved measurement data
        unobserved_data = np.column_stack((unobserved_measurements
                                           ,np.full(unobserved_measurements.shape[0],
                                                    'no signal')))
        # Combine into measurement data
        measurement_data = np.vstack((observed_data, unobserved_data))

        # Prepare noise emitters data
        noise_data = np.column_stack((noise_emitters
                                      ,np.full(noise_emitters.shape[0], 'noise')))

        # Combine all data into a single array
        all_data = np.vstack((ground_truth, measurement_data, noise_data))

        # Create header for the CSV file
        header = 'x,y,z,type\n'

        # Save to CSV
        np.savetxt(filename, all_data, delimiter=',', header=header, comments='', fmt='%s')

        print(f"Data saved to {filename}")

    elif filename.endswith('.h5'):
        # Save to HDF5
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('true_emitters', data=true_emitters)
            hf.create_dataset('unlabelled_emitters', data=unlabelled_emitters)
            hf.create_dataset('observed_measurements', data=observed_measurements)
            hf.create_dataset('unobserved_measurements', data=unobserved_measurements)
            hf.create_dataset('noise_emitters', data=noise_emitters)
        print(f"Data saved to {filename}")

    else:
        print("Unsupported file format. Please use .csv or .h5")