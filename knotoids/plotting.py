from typing import Dict, List, Tuple

import grapher
import numpy as np
import plotly.graph_objects as go


def plot_from_nodes_and_edges(
    nodes: Dict[int, grapher.SphericalNode], edges: List[Tuple[int, int]]
) -> None:
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
