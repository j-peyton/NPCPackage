import numpy as np

def gen_noise(xrange, yrange, noise_density):
    """
    Generates random noise emitters across the specified window area (2D or 3D).

    :param xrange: Tuple specifying the (min, max) x-coordinate bounds for the window.
    :param yrange: Tuple specifying the (min, max) y-coordinate bounds for the window.
    :param noise_density: Average number of noise emitters per unit area (for 2D) or volume (for 3D).

    :return: Array of noise emitter coordinates (2D or 3D).
    """
    area = (xrange[1] - xrange[0]) * (yrange[1] - yrange[0])
    n_noise_emitters = np.random.poisson(noise_density * area)

    # Generate the 2D noise for x and y coordinates
    x_noise = np.random.uniform(xrange[0], xrange[1], n_noise_emitters)
    y_noise = np.random.uniform(yrange[0], yrange[1], n_noise_emitters)

    # If a membrane function is provided, make noise 3D
    clutter = np.column_stack((x_noise, y_noise))

    return clutter
