import numpy as np

def gen_dimer(radius):
    """"
    Generates a dimer structure around a point.

    :param radius: The radius of the structure.
    :return: Numpy array detailing the dimer coordinates.
    """
    # Generate a random angle between 0 and 2*pi
    angle = np.random.uniform(0, 2 * np.pi)

    # Calculate the positions of the two emitters
    emitter1 = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    emitter2 = -emitter1

    return np.array([emitter1, emitter2])


def gen_poly(n, radius=1):
    """"
    Generates a regular polygonal structure around a point.

    :param n: Number of sides of the polygon.
    :param radius: The radius of the structure.
    :return: Numpy array detailing the vertices of the polygon.
    """
    if n < 3:
        raise ValueError("n must be greater than or equal to 3")

    # Generate x, y coordinates of the polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    coords = np.column_stack((x, y))

    theta = np.random.uniform(0, 2 * np.pi)
    # 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    rotated_poly = coords @ rotation_matrix.T

    return rotated_poly


def gen_preset(structure, radius, sides=None):
    """
    Generates a preset structure.

    :param structure: Type of the structure ('dimer' or 'polygon').
    :param radius: Radius of the structure.
    :param sides: Number of sides for polygons (ignored for dimers).

    :return: Numpy array of the structure's coordinates.
    """
    if sides is None:
        sides = np.random.choice(range(3,9))
    if structure == 'dimer':
        return gen_dimer(radius)
    elif structure == 'polygon':
        if sides is None:
            raise ValueError("Number of sides must be specified")
        return gen_poly(sides, radius)
    else:
        raise ValueError(f"Unknown structure input: {structure}")
