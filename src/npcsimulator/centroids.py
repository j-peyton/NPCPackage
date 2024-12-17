import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import time

def generate_centroids(xrange, yrange, poisson_mean, radius):
    """
    Generates centroids via a Poisson process across a specified window, with hard-core distance.
    Uses KDTree to check the distance between points efficiently.

    :param xrange: Tuple (min, max), specify the window in which centroids are generated for x.
    :param yrange: Tuple (min, max), specify the window in which centroids are generated for y.
    :param poisson_mean: Numeric, centroids are generated with Poisson(poisson_mean).
    :param radius: Radius of the centroid/emitter structures, enforces hard-core distance.

    :return: Numpy array of centroids.
    """
    centroids = []  # List to store valid centroids

    # Ensure a realistic number of centroids given constraints
    x_min, x_max = xrange
    y_min, y_max = yrange
    max_points = int(int((x_max - x_min) * (y_max - y_min) / (np.pi * (radius ** 2))) / 2)
    poisson_points = np.random.poisson(poisson_mean)
    npoints = min(poisson_points, max_points)

    start_time = time.time()


    while len(centroids) < npoints:
        # Generate a candidate point
        x = np.random.uniform(xrange[0], xrange[1])
        y = np.random.uniform(yrange[0], yrange[1])
        candidate = np.array([x, y])

        if len(centroids) > 0:
            # Build KDTree with existing centroids
            tree = KDTree(centroids)
            # Check the distance to all existing centroids
            distances = tree.query(candidate, k=1)[0]
            if distances >= 2 * radius:
                centroids.append(candidate)
        else:
            # If no centroids exist, accept this candidate
            centroids.append(candidate)

        elapsed_time = time.time() - start_time
        if elapsed_time > 15:
            print("Centroid generation taking a long time. Consider reducing poisson_mean.")
            break

    return np.array(centroids)





def read_centroids(filename, radius=None):
    """
    Reads centroids from a CSV file and filters them based on the steric distance.
    Supports updating z-coordinates using membrane_function if provided.

    :param filename: Path to the CSV file.
    :param radius: Radius of the centroid/emitter structures, enforces hard-core distance.
    :return: Numpy array of filtered centroids.
    """
    # Read the CSV file with fixed column names, skip the header row
    data = pd.read_csv(filename, skiprows=1, header=None)

    # Determine if the data contains 'z' column
    if len(data.columns) == 3:
        data.columns = ['x', 'y', 'z']  # 3D case
        centroids = data[['x', 'y', 'z']].to_numpy()
    elif len(data.columns) == 2:
        data.columns = ['x', 'y']  # 2D case
        centroids = data[['x', 'y']].to_numpy()
    else:
        raise ValueError(f"Centroids must be 2D or 3D, instead received {len(data.columns)}")

    print(f"Successfully loaded centroids from {filename}.")

    tree = KDTree(centroids)

    # If radius is none, calculate it from the smallest distance
    if radius is None:
        distances, _ = tree.query(centroids, k=2)  # k=2 to get the nearest neighbor
        radius = 0.5*np.min(distances[:, 1])

    # Filter centroids based on the radius
    filtered = [centroids[0]]

    for centroid in centroids[1:]:
        # Check distances between the current centroid and all filtered centroids
        filtered_np = np.array(filtered)
        distances = np.linalg.norm(filtered_np - centroid, axis=1)  # Calculate Euclidean distance

        if np.all(distances > 2*radius):
            filtered.append(centroid)

    return np.array(filtered)


def get_range(filepath):
    """
    Read centroids from a file and return their ranges.
    If the file contains 'x', and 'y' columns, it returns ranges for them.

    :param filepath: Path to the input file.
    :return: A tuple (xrange, yrange) representing the min and max ranges of x and y.
    """
    try:
        # Read the file
        data = pd.read_csv(filepath, skiprows=1)

        # Ensure the file contains 'x' and 'y' columns
        if 'x' not in data.columns or 'y' not in data.columns:
            raise ValueError("The file must contain 'x' and 'y' columns.")

        # Calculate xrange and yrange
        xrange = [data['x'].min(), data['x'].max()]
        yrange = [data['y'].min(), data['y'].max()]

        return xrange, yrange
    except Exception as e:
        print(f"Error reading the centroid file: {e}")
        raise