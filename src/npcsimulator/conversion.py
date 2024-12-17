import numpy as np

def convert_3d(array_list, membrane_function):
    """
    Converts a list of 2D arrays into 3D using the membrane function input.

    :param array_list: List of Nx2 2D arrays.
    :param membrane_function: Cell membrane function used to generate the z-coordinates.

    :return: List of Nx3 3D arrays.
    """
    arrays_3d = []

    for array in array_list:
        if array.shape[1] != 2:
            raise ValueError("Input arrays must be two dimensional.")

        z_coords = membrane_function(array[:, 0], array[:, 1])

        array_3d = np.column_stack((array, z_coords))
        arrays_3d.append(array_3d)

    return arrays_3d