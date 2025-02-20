import plotly.graph_objs as go
import numpy as np

def plot_components_in3d(centroids, labelled_emitters, unlabelled_emitters,
                            observed_measurements, noise=None):
    """
    Plots all output components of dist_custom (labelled emitters, unlabelled emitters, measurements)
    The plot is 3D or 2D based on the data.

    :param labelled_emitters: Numpy array of labelled emitter coordinates.
    :param unlabelled_emitters: Numpy array of unlabelled emitter coordinates.
    :param observed_measurements: Numpy array of observed measurements.
    :param centroids: Numpy array of the centroid coordinates (to check for 3D/2D).
    :param noise: Numpy array of clutter.
    """

    # Check if the data is 3D
    is_3d = labelled_emitters.shape[1] == 3

    # Initialize the plot data
    fig = go.Figure()

    # Plot centroids (in both 2D or 3D)
    if centroids is not None:
        fig.add_trace(go.Scatter3d(
           x=centroids[:, 0],
           y=centroids[:, 1],
           z=centroids[:, 2] if is_3d else np.zeros_like(centroids[:, 0]),
           mode='markers',
           marker=dict(size=6, color='purple', opacity=1),
           name='Centroids'
       ))

    # Plot labelled emitters
    fig.add_trace(go.Scatter3d(
        x=labelled_emitters[:, 0],
        y=labelled_emitters[:, 1],
        z=labelled_emitters[:, 2] if is_3d else np.zeros_like(labelled_emitters[:, 0]),
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.8),
        name='Labelled Emitters'
    ))

    # Plot unlabelled emitters
    if unlabelled_emitters is not None:
        fig.add_trace(go.Scatter3d(
           x=unlabelled_emitters[:, 0],
           y=unlabelled_emitters[:, 1],
           z=unlabelled_emitters[:, 2] if is_3d else np.zeros_like(unlabelled_emitters[:, 0]),
           mode='markers',
           marker=dict(size=4, color='blue', opacity=0.8),
           name='Unlabelled Emitters'
       ))

    # Plot observed measurements
    fig.add_trace(go.Scatter3d(
        x=observed_measurements[:, 0],
        y=observed_measurements[:, 1],
        z=observed_measurements[:, 2] if is_3d else np.zeros_like(observed_measurements[:, 0]),
        mode='markers',
        marker=dict(size=3, color='orange', opacity=0.7),
        name='Observed Measurements'
    ))

    # Plot noise
    if noise is not None:
        fig.add_trace(go.Scatter3d(
           x=noise[:, 0],
           y=noise[:, 1],
           z=noise[:, 2] if is_3d else np.zeros_like(noise[:, 0]),
           mode='markers',
           marker=dict(size=3, color='yellow', opacity=0.4),
           name='Clutter'
       ))

    # Set layout for 3D/2D plots
    z_values = []

    if centroids is not None:
        z_values.extend(centroids[:, 2])
    if labelled_emitters is not None:
        z_values.extend(labelled_emitters[:, 2])
    if unlabelled_emitters is not None:
        z_values.extend(unlabelled_emitters[:, 2])
    if observed_measurements is not None:
        z_values.extend(observed_measurements[:, 2])
    if noise is not None:
        z_values.extend(noise[:, 2])

    if z_values:  # Ensure there are valid z-values
        zmin = min(z_values)
        zmax = max(z_values)
        zrange = [zmin - 2.5, zmax + 2.5]
    else:
        zrange = None  # No z-values, default Plotly behavior

    if is_3d:
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis=dict(title='Z', range=zrange) # Update this as needed
            ),
            title="3D Distribution of Emitters and Measurements",
            showlegend=True
        )
    else:
        fig.update_layout(
            title="2D Distribution of Emitters and Measurements",
            showlegend=True
        )

    # Show the plot
    fig.show()


def plot_components_scale(centroids, labelled_emitters, unlabelled_emitters,
                    observed_measurements, noise=None):
    """
    Plots all output components of dist_custom (labelled emitters, unlabelled emitters, measurements)
    The plot is 2D, with the Z-axis values represented as a color scale if the data is 3D.

    :param centroids: Numpy array of the centroid coordinates (to check for 3D/2D).
    :param labelled_emitters: Numpy array of labelled emitter coordinates.
    :param unlabelled_emitters: Numpy array of unlabelled emitter coordinates.
    :param observed_measurements: Numpy array of observed measurements.
    :param noise: Numpy array of clutter (optional).
    """

    # Check if the data is 3D
    is_3d = centroids.shape[1] == 3

    # Initialize the plot data
    fig = go.Figure()

    # Plot centroids (in both 2D or 3D)
    if is_3d:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(size=4, color=centroids[:, 2], colorscale='Viridis', colorbar=dict(title="Z Value")),
            name='Centroids'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers',
            marker=dict(size=4, color='purple', opacity=1),
            name='Centroids'
        ))

    # Plot labelled emitters
    if is_3d:
        fig.add_trace(go.Scatter(
            x=labelled_emitters[:, 0],
            y=labelled_emitters[:, 1],
            mode='markers',
            marker=dict(size=4, color=labelled_emitters[:, 2], colorscale='Reds', colorbar=dict(title="Z Value")),
            name='Labelled Emitters'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=labelled_emitters[:, 0],
            y=labelled_emitters[:, 1],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.8),
            name='Labelled Emitters'
        ))

    # Plot unlabelled emitters
    if is_3d:
        fig.add_trace(go.Scatter(
            x=unlabelled_emitters[:, 0],
            y=unlabelled_emitters[:, 1],
            mode='markers',
            marker=dict(size=4, color=unlabelled_emitters[:, 2], colorscale='Blues', colorbar=dict(title="Z Value")),
            name='Unlabelled Emitters'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=unlabelled_emitters[:, 0],
            y=unlabelled_emitters[:, 1],
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.8),
            name='Unlabelled Emitters'
        ))

    # Plot observed measurements
    if is_3d:
        fig.add_trace(go.Scatter(
            x=observed_measurements[:, 0],
            y=observed_measurements[:, 1],
            mode='markers',
            marker=dict(size=3, color=observed_measurements[:, 2], colorscale='Oranges', colorbar=dict(title="Z Value")),
            name='Observed Measurements'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=observed_measurements[:, 0],
            y=observed_measurements[:, 1],
            mode='markers',
            marker=dict(size=3, color='orange', opacity=0.7),
            name='Observed Measurements'
        ))

    # Plot noise
    if noise is not None:
        if is_3d:
            fig.add_trace(go.Scatter(
                x=noise[:, 0],
                y=noise[:, 1],
                mode='markers',
                marker=dict(size=3, color=noise[:, 2], colorscale='YlGnBu', colorbar=dict(title="Z Value")),
                name='Clutter'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=noise[:, 0],
                y=noise[:, 1],
                mode='markers',
                marker=dict(size=3, color='yellow', opacity=0.4),
                name='Clutter'
            ))

    # Ensure square aspect ratio by using scaleanchor
    fig.update_layout(
        title="2D Distribution of Emitters and Measurements",
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True,
        legend=dict(
            x=-0.1,  # Position the legend to the left
            y=1,  # Keep it aligned to the top
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: Transparent background
            bordercolor='Black',
            borderwidth=2
        ),
        xaxis=dict(scaleanchor='y'),  # Ensures square aspect ratio for x and y
        yaxis=dict(scaleanchor='x')   # Ensures square aspect ratio for x and y
    )

    # Show the plot
    fig.show()
