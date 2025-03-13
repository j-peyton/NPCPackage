import numpy as np
import h5py

def generate_measurements(emitter_position, poisson_mean, uncertainty_std):
    """
    Generates repeated measurements for each emitter, with uncertainty, optionally applying a membrane function.

    :param emitter_position: Numpy array of the emitter's true positions (N x 2 or N x 3).
    :param poisson_mean: Mean of the Poisson distribution for the number of measurements.
    :param uncertainty_std: Standard deviation of the Gaussian uncertainty.

    :return: Numpy array of repeated measurements (M x 2 or M x 3).
    """
    emitter_position = np.atleast_2d(emitter_position)  # Ensure input is at least 2D
    new_emits = []

    for pos in emitter_position:
        # Generate random offsets
        num_measurements = np.random.poisson(poisson_mean)
        offsets = np.random.normal(loc=0, scale=uncertainty_std, size=(num_measurements, 2))
        measurements = pos + offsets
        new_emits.extend(measurements)

    return np.array(new_emits)


def gen_noise(xrange, yrange, rho, measured=5, ms_uncertainty=0.5):
    n_noise_emitters = np.random.poisson(rho)
    x_noise = np.random.uniform(xrange[0], xrange[1], n_noise_emitters)
    y_noise = np.random.uniform(yrange[0], yrange[1], n_noise_emitters)
    noise = np.column_stack((x_noise, y_noise))
    clutter = []
    for point in noise:
        measurements = generate_measurements(point, poisson_mean=measured, uncertainty_std=ms_uncertainty)
        for measurement in measurements:
            clutter.append((measurement[0], measurement[1], 0))  # ID is always 0 for clutter
    return np.array(clutter)


def convert_3d(array_list, membrane_function):
    """
    Converts a list of 2D arrays into 3D using the membrane function input.

    :param array_list: List of Nx2 2D arrays.
    :param membrane_function: Cell membrane function used to generate the z-coordinates.

    :return: List of Nx3 3D arrays.
    """
    arrays_3d = []

    for array in array_list:
        # Ensure that the array is 2D with shape (N, 2), if not, raise an error
        if array.ndim == 1:
            array = array.reshape(-1, 2)  # Reshape 1D to 2D (e.g., [x, y] -> [[x, y]])
        elif array.shape[1] != 2:
            raise ValueError("Input arrays must have shape (N, 2) for x and y coordinates.")

        # Extract the x and y coordinates
        x_coords, y_coords = array[:, 0], array[:, 1]

        # Generate the z-coordinates using the membrane function
        z_coords = membrane_function(x_coords, y_coords)

        # Stack the x, y, z coordinates into a 3D array
        array_3d = np.column_stack((array, z_coords))
        arrays_3d.append(array_3d)

    return arrays_3d



def dist_custom(filename, centroids, p, q, radius, structures, abundances, gt_uncertainty=0,
                measured=7, ms_uncertainty=0.05, noise_params=None):
    observed, edges, emitter_data = [], [], []
    emitter_index = 1

    abundances = np.array(abundances) / np.sum(abundances)

    if not structures or len(structures) == 0:
        raise ValueError("Structures list is empty or None.")
    if abundances is None or len(abundances) == 0:
        raise ValueError("Abundances list is empty or None.")
    if len(structures) != len(abundances):
        raise ValueError("Number of structures and abundances must match.")

    for centroid in centroids:
        structure_idx = np.random.choice(len(structures), p=abundances)
        structure = radius * structures[structure_idx]

        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        rotated_structure = (structure @ rotation_matrix.T) + centroid
        emitter_indices = []

        for point in rotated_structure:
            corrected_point = np.random.normal(loc=point, scale=gt_uncertainty)
            emitter_type = "labelled" if np.random.binomial(1, p) else "unlabelled"

            emitter_data.append((emitter_index, corrected_point[0], corrected_point[1], emitter_type))
            emitter_indices.append(emitter_index)
            emitter_index += 1

            if emitter_type == "labelled":
                measurements = generate_measurements(corrected_point, poisson_mean=measured,
                                                     uncertainty_std=ms_uncertainty * radius)
                for measurement in measurements:
                    if np.random.binomial(1, q):
                        observed.append((measurement[0], measurement[1], emitter_indices[-1]))

        for i in range(len(emitter_indices)):
            for j in range(i + 1, len(emitter_indices)):
                edges.append((emitter_indices[i], emitter_indices[j]))

    clutter_data = []
    if noise_params:
        xrange, yrange, rho= noise_params
        clutter_data = gen_noise(xrange, yrange, rho, measured, ms_uncertainty)
        observed.extend(clutter_data)

    # Save data to HDF5 file
    with h5py.File(filename, 'w') as hf:
        emitter_group = hf.create_group('emitter')
        emitter_group.create_dataset('id', data=np.array([e[0] for e in emitter_data], dtype=np.int32))
        emitter_group.create_dataset('position', data=np.array([[e[1], e[2]] for e in emitter_data]))
        emitter_group.create_dataset('type', data=np.array([e[3] for e in emitter_data], dtype='S'))

        if observed:
            observed_group = hf.create_group('observed')
            observed_group.create_dataset('position', data=np.array([[o[0], o[1]] for o in observed]))
            observed_group.create_dataset('emitter_id', data=np.array([o[2] for o in observed], dtype=np.int32))

        clutter_group = hf.create_group('clutter')
        clutter_group.create_dataset('position', data=np.array([[c[0], c[1]] for c in clutter_data]))
        clutter_group.create_dataset('emitter_id', data=np.array([c[2] for c in clutter_data], dtype=np.int32))
        clutter_group.create_dataset('type', data=np.array(['clutter'] * len(clutter_data), dtype='S'))

        hf.create_dataset('edges', data=np.array(edges, dtype=np.int32))