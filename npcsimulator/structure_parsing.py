import numpy as np
from .templates import gen_dimer, gen_poly


def parse_custom(x_structures, y_structures):
    """
    Parses custom 2D structures based on provided coordinates.

    :param x_structures: List of lists containing x-coordinates for custom structures.
    :param y_structures: List of lists containing y-coordinates for custom structures.

    :return: List of numpy arrays representing custom 2D structures.
    """
    if len(x_structures) != len(y_structures):
        raise ValueError("x_structures and y_structures must have the same length.")

    # Combine x and y coordinates into 2D structures
    return [np.column_stack((x, y)) for x, y in zip(x_structures, y_structures)]


def parse_structures(x_structures=None, y_structures=None,
                     preset_structures=None, preset_sides=None, structure_abundance=None):
    """
    Parses custom and preset 2D structures into a unified list with specified abundances.

    :param x_structures: List of lists containing x-coordinates for custom structures.
    :param y_structures: List of lists containing y-coordinates for custom structures.
    :param preset_structures: List of names of preset structures (e.g., 'polygon', 'dimer').
    :param preset_sides: List of polygon side counts (used only for 'polygon').
    :param structure_abundance: List of abundances for all structures (custom and preset).

    :return: A tuple of (all_structures, normalized_abundances).
    """
    # Generate preset structures if provided
    preset_structure_list = []
    if preset_structures:
        for structure, sides in zip(preset_structures, preset_sides or []):
            if structure == 'polygon':
                preset_structure_list.append(gen_poly(sides, radius=1))  # Generate polygon
            elif structure == 'dimer':
                preset_structure_list.append(gen_dimer(radius=1))  # Generate dimer
            else:
                raise ValueError(f"Unknown preset structure: {structure}")

    # Parse custom structures into 2D
    if x_structures is not None:
        custom_structures = parse_custom(x_structures, y_structures)
        all_structures = preset_structure_list + custom_structures
    else:
        all_structures = preset_structure_list

    # Validate and normalize abundances
    if structure_abundance is None or len(all_structures) != len(structure_abundance):
        raise ValueError("Number of abundances must match the number of structures")

    abundances = np.array(structure_abundance, dtype=np.float64)
    abundances /= abundances.sum()

    return all_structures, abundances