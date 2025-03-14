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
            clutter.append((measurement[0], measurement[1], -1))  # ID is always -1 for clutter
    return np.array(clutter)



def dist_custom(filename, centroids, p, q, radius, structures, abundances, gt_uncertainty=0,
                measured=7, ms_uncertainty=0.05, noise_params=None, membrane_function=None):
    observed_data, edges, emitter_data = [], [], []
    emitter_index = 0

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

            emitter_data.append((corrected_point[0], corrected_point[1], emitter_index, emitter_type))
            emitter_indices.append(emitter_index)
            emitter_index += 1

            if emitter_type == "labelled":
                measurements = generate_measurements(corrected_point, poisson_mean=measured,
                                                     uncertainty_std=ms_uncertainty * radius)
                for measurement in measurements:
                    if np.random.binomial(1, q):
                        observed_data.append((measurement[0], measurement[1], emitter_indices[-1]))

        for i in range(len(emitter_indices)):
            for j in range(i + 1, len(emitter_indices)):
                edges.append((emitter_indices[i], emitter_indices[j]))

    clutter_data = []
    if noise_params:
        xrange, yrange, rho= noise_params
        clutter_data = gen_noise(xrange, yrange, rho, measured, ms_uncertainty)

    # Create emitter_pos, observed_pos arrays for easier 3d processing
    emitter_pos = np.array([[e[0], e[1]] for e in emitter_data])
    observed_pos = np.array([[o[0], o[1]]for o in observed_data])

    print("Data Types Before Saving:")
    print("Emitter Data:", type(emitter_pos), "Length:", len(emitter_pos))
    print("Observed Data:", type(observed_pos), "Length:", len(observed_pos))
    print("Clutter Data:", type(clutter_data), "Length:", len(clutter_data))
    print("Edges:", type(edges), "Length:", len(edges))


    print("Emitter Position Shape:", np.array([e[:-1] for e in emitter_data]).shape)

    print("Observed Position Shape:", np.array([o[:-1] for o in observed_data]).shape)

    print("Clutter Position Shape:", np.array([c[:-1] for c in clutter_data]).shape)

    # Save data to HDF5 file
    with h5py.File(filename, 'w') as hf:
        emitter_group = hf.create_group('emitter')
        emitter_group.create_dataset('id', data=np.array([e[2] for e in emitter_data], dtype=np.int32))
        emitter_group.create_dataset('position', data=np.array([[e[0], e[1]] for e in emitter_data]))
        emitter_group.create_dataset('type', data=np.array([e[3] for e in emitter_data], dtype='S'))

        if observed_data:
            observed_group = hf.create_group('observed')
            observed_group.create_dataset('position', data=np.array([[o[0], o[1]] for o in observed_data]))
            observed_group.create_dataset('emitter_id', data=np.array([o[2] for o in observed_data], dtype=np.int32))

        clutter_group = hf.create_group('clutter')
        clutter_group.create_dataset('position', data=np.array([[c[0], c[1]] for c in clutter_data]))
        clutter_group.create_dataset('emitter_id', data=np.array([c[2] for c in clutter_data], dtype=np.int32))
        clutter_group.create_dataset('type', data=np.array(['clutter'] * len(clutter_data), dtype='S'))

        hf.create_dataset('edges', data=np.array(edges, dtype=np.int32))