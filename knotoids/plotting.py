from typing import Dict, List, Tuple

import grapher
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from structures import (
    Edge,
    PlanarNode,
    PlanarNodeDict,
    SphericalNode,
    SphericalNodeDict,
)


def plot_from_nodes_and_edges(nodes: SphericalNodeDict, edges: List[Edge]) -> None:
    points = []
    for v in nodes:
        points.append(nodes[v].position)
    points = np.asarray(points)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color="red",
            ),
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
