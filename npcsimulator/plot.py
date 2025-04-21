import h5py
import plotly.express as px
import pandas as pd


def plot_h5(filename: str):
    """
    Load and plot emitter, observed, and clutter data from an HDF5 file.

    Args:
        filename: Path to the HDF5 file containing simulation output.

    Returns:
        Displays an interactive Plotly 3D scatter plot.
    """
    data = []

    with h5py.File(filename, 'r') as f:
        if 'emitter' in f:
            positions = f['emitter']['position'][:]
            types = f['emitter']['type'][:].astype(str)
            for pos, t in zip(positions, types):
                data.append((pos[0], pos[1], pos[2], t))

        if 'observed' in f:
            positions = f['observed']['position'][:]
            for pos in positions:
                data.append((pos[0], pos[1], pos[2], 'observed'))

        if 'clutter' in f:
            positions = f['clutter']['position'][:]
            for pos in positions:
                data.append((pos[0], pos[1], pos[2], 'clutter'))

    df = pd.DataFrame(data, columns=["x", "y", "z", "type"])

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="type",
                        symbol="type", title="Simulated Dataset Visualization",
                        opacity=0.7, width=800, height=600)
    fig.update_layout(legend_title_text='Data Type', legend=dict(x=0.02, y=0.98))
    fig.show()
