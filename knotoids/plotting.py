from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from spherical_geometry import great_circle_arc
from structures import Edge, PlanarNodeDict, Region, SphericalNodeDict


def plot_spherical_regions(
    regions: List[Region], curve: Optional[np.ndarray] = None
) -> None:
    fig = go.Figure()
    colours = px.colors.qualitative.Dark24
    for idx, region in enumerate(regions):
        colour = colours[idx % 24]
        points = np.array(
            [node.position for node in region.boundary_nodes]
            + [region.boundary_nodes[0].position]
        )
        representative = region.internal_point
        interpolated = _interpolate_great_arcs(points)
        fig.add_trace(
            go.Scatter3d(
                x=interpolated[:, 0],
                y=interpolated[:, 1],
                z=interpolated[:, 2],
                marker=dict(
                    size=1,
                    color=colour,
                ),
                line=dict(color=colour, width=3),
                hoverinfo="text",
                hovertemplate=idx,
            )
        )
        # TODO: shade region. Implementation below isn't great.
        # fig.add_trace(
        #     go.Mesh3d(
        #         x=interpolated[:, 0],
        #         y=interpolated[:, 1],
        #         z=interpolated[:, 2],
        #         opacity=0.5,
        #         color=colour,  # or any color you prefer
        #     )
        # )
        # plot representative point
        fig.add_trace(
            go.Scatter3d(
                x=[representative[0]],
                y=[representative[1]],
                z=[representative[2]],
                marker=dict(
                    size=3,
                    color=colour,
                ),
                mode="markers",
                hoverinfo="text",
                hovertemplate=idx,
            )
        )
    if curve is not None:
        curve = _rescale_curve(curve)
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                marker=dict(
                    size=1,
                    color="black",
                ),
                line=dict(color="black", width=2),
            )
        )
    fig.update_layout(
        dict(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=["visible", "legendonly"],
                                label="Deselect All",
                                method="restyle",
                            ),
                            dict(
                                args=["visible", True],
                                label="Select All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top",
                ),
            ]
        )
    )
    fig.show()


def plot_spherical_from_nodes_and_edges(
    nodes: SphericalNodeDict, edges: List[Edge], curve: Optional[np.ndarray] = None
) -> None:
    points = np.asarray([nodes[v].position for v in nodes])
    hover_texts = [v.index for v in nodes.values()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=hover_texts,  # Coloring by index
                colorscale="Viridis",  # Change to your preferred colorscale if needed
                colorbar=dict(title="Node Index"),
                showscale=True,
            ),
            text=hover_texts,  # Assign hover text to each node
            hoverinfo="text",
            line=dict(color="black", width=2),
        )
    )
    for e in edges:
        s, t = e
        p = nodes[s].position
        q = nodes[t].position
        obj = go.Scatter3d(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            z=[p[2], q[2]],
            mode="lines",
            marker=dict(
                size=1,
                color="black",
            ),
            line=dict(color="black", width=2),
        )
        fig.add_trace(obj)
    if curve is not None:
        curve = _rescale_curve(curve)
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                marker=dict(
                    size=1,
                    color="black",
                ),
                line=dict(color="black", width=2),
            )
        )
    fig.show()


def plot_planar_from_nodes_and_edges(nodes: PlanarNodeDict, edges: List[Edge]) -> None:
    x, y, indices = [], [], []

    for _, v in nodes.items():
        x.append(v.position[0])
        y.append(v.position[1])
        indices.append(v.index)

    fig = px.scatter(x=x, y=y)
    fig.update_traces(hoverinfo="text", hovertemplate=indices)

    for e in edges:
        s, t = e
        p = nodes[s].position
        q = nodes[t].position
        obj = go.Scatter(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            mode="lines",
            marker=dict(
                size=2,
                color="black",
            ),
            line=dict(color="black", width=2),
        )
        fig.add_trace(obj)

    fig.show()


def _interpolate_great_arcs(points: np.ndarray) -> np.ndarray:
    """
    Returns array of points with each arc interpolated.
    """
    interpolated = []
    n = points.shape[0]
    for i in range(n - 1):
        length = great_circle_arc.length(points[i, :], points[i + 1, :])
        steps = max((np.ceil(length / 2), 2))
        arc_points = great_circle_arc.interpolate(
            points[i, :], points[i + 1, :], steps=steps
        )
        interpolated.append(arc_points)
    return np.vstack(interpolated)


def _rescale_curve(curve: np.ndarray):
    """
    Rescale curve to fit into unit sphere (for plotting only).
    """
    curve = curve - curve.mean(axis=0)
    norms = np.linalg.norm(curve, axis=1)
    return curve / (np.max(norms) * 2)
