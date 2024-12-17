import argparse
from pathlib import Path
import importlib.util

from .centroids import generate_centroids, read_centroids, get_range
from .structure_parsing import parse_structures
from .emitters import dist_custom, gen_noise
from .conversion import convert_3d
from .plot import plot_components_in3d
from .save import save_data

def load_custom_function(filepath):
    """Load a custom membrane function from a file."""
    module_path = Path(filepath)
    module_name = module_path.stem  # Get the file name without the extension
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.membrane_function

def cli():
    parser = argparse.ArgumentParser(description="Simulate emitter distribution around centroids.")

    parser.add_argument('--xrange', type=float, nargs=2, default=[-50, 50],
                        help='X coordinate range for centroids')
    parser.add_argument('--yrange', type=float, nargs=2, default=[-50, 50],
                        help='Y coordinate range for centroids')
    parser.add_argument('--poisson_mean', type=int, default=15,
                        help='Generates centroids via poisson distribution defined by this variable')
    parser.add_argument('--centroids', type=str, default=None,
                        help='Path to a csv file containing centroids')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='Centroid/emitter structure radius.')
    parser.add_argument('--Pe', type=float, default=0.80,
                        help='Probability of successful labelling')
    parser.add_argument('--Pf', type=float, default=0.80,
                        help='Probability of fluorescence / signal received')
    parser.add_argument('--gt_uncertainty', type=float, default=0.1,
                        help="Uncertainty of the position of the ground truth emitter")
    parser.add_argument('--measured', type=int, default=5,
                        help='Poisson mean of the number of measurements per emitter')
    parser.add_argument('--ms_uncertainty', type=float, default=0.05,
                        help='Uncertainty of repeated measurements around an emitter as a percentage'
                             'of the radius of the structure')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Average number of noise emitters per unit area')
    parser.add_argument('--output', type=str, default=None,
                        help="Output file name with .csv or .h5 extension")
    parser.add_argument('--x_structures', type=float, nargs='+', action='append', default=None,
                        help='List of x coordinates for each structure. Specify multiple structures as separate lists.'
                             'E.g. --x_structures 2 4 5 6 --x_structures 4 9 1 4')
    parser.add_argument('--y_structures', type=float, nargs='+', action='append', default=None,
                        help='List of y coordinates for each structure. Specify multiple structures as separate lists.'
                             'E.g. --y_structures 1 3 5 7 --x_structures 2 5 3 8')
    parser.add_argument('--structure_abundances', type=float, nargs='+',
                        help='Abundance values for each structure. '
                             'E.g. --structure_abundance 0.8 0.2')
    parser.add_argument('--preset_structures', nargs='+', choices=['dimer','polygon'], default=None,
                        help="Preset structure to use in emitter distribution")
    parser.add_argument('--preset_sides', type=int, nargs='+', default=[3,4,5,6,7,8],
                        help="List of side counts for preset polygons")
    parser.add_argument('--membrane', type=str,
                        help="Filepath to the custom membrane function. Store membrane function in its own isolated .py file")

    args = parser.parse_args()

    if args.centroids:
        print(f"Reading centroids from {args.centroids}...")
        centroids = read_centroids(args.centroids, args.radius)
        args.xrange, args.yrange = get_range(args.centroids)
    else:
        print("Generating centroids via Poisson process...")
        centroids = generate_centroids(args.xrange, args.yrange, args.poisson_mean, args.radius)

    if args.x_structures or args.preset_structures:
        if not args.structure_abundances:
            raise ValueError("Structure abundances must be specified.")

        print(f"X Structures: {args.x_structures}")
        print(f"Y Structures: {args.y_structures}")
        print(f"Preset Structures: {args.preset_structures}")
        print(f"Preset Sides: {args.preset_sides}")
        print(f"Structure Abundance: {args.structure_abundances}")

        # Parse all structures and validate abundances
    all_structures, abundances = parse_structures(
            x_structures=args.x_structures, y_structures=args.y_structures,
            preset_structures=args.preset_structures, preset_sides=args.preset_sides,
            structure_abundance=args.structure_abundances
        )

    (labelled_emits, unlabelled_emits, observed_measurements,
         unobserved_measurements) = dist_custom(
            centroids, args.Pe, args.Pf, radius=args.radius, gt_uncertainty=args.gt_uncertainty,
        measured=args.measured, ms_uncertainty=args.ms_uncertainty, structures=all_structures,
        abundances=abundances
        )

    # Generate random noise emitters
    noise_emits = gen_noise(args.xrange, args.yrange, args.noise)

    if args.membrane:
        membrane_path = Path(args.membrane).resolve()
        membrane_function = load_custom_function(membrane_path)
        array_list = [centroids, labelled_emits, unlabelled_emits, observed_measurements,
                      unobserved_measurements, noise_emits]
        array_list = convert_3d(array_list, membrane_function)
        (centroids, labelled_emits, unlabelled_emits, observed_measurements,
         unobserved_measurements, noise_emits) = array_list

    else:
        (centroids, labelled_emits, unlabelled_emits, observed_measurements,
         unobserved_measurements, noise_emits) = (centroids, labelled_emits, unlabelled_emits, observed_measurements,
                      unobserved_measurements, noise_emits)

    plot_components_in3d(centroids=centroids, labelled_emitters=labelled_emits,
                         unlabelled_emitters=unlabelled_emits, observed_measurements=observed_measurements,
                         unobserved_measurements=unobserved_measurements, noise=noise_emits)

    if args.output:
        save_data(args.output, labelled_emits, unlabelled_emits, observed_measurements,
                  unobserved_measurements, noise_emits)
    else:
        print("No output specified; data not saved")

if __name__ == '__main__':
    cli()