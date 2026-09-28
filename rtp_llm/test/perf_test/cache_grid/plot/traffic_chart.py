"""Common Plotly rendering for production traffic heatmaps."""

import math


def render_traffic_chart(
    points,
    output,
    *,
    title,
    z_title,
    trace_name,
    hovertemplate,
    custom_fields,
    marker_size=3.0,
    log_z=False,
    log_color=False,
    color_min_percentile=0.0,
    color_max_percentile=100.0,
):
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error
    if not 0 <= color_min_percentile <= color_max_percentile <= 100:
        raise ValueError("color percentiles must satisfy 0 <= min <= max <= 100")
    colors = [math.log1p(p["count"]) if log_color else p["count"] for p in points]
    if colors and (color_min_percentile > 0 or color_max_percentile < 100):
        ordered = sorted(colors)
        n = len(ordered)
        lower = ordered[min(n - 1, int(n * color_min_percentile / 100))]
        upper = ordered[max(0, min(n - 1, int(n * color_max_percentile / 100) - 1))]
        colors = [max(lower, min(upper, value)) for value in colors]
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=[p["x"] for p in points],
                y=[p["y"] for p in points],
                z=[p["z"] for p in points],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": colors,
                    "colorscale": "Hot",
                    "reversescale": True,
                    "opacity": 0.8,
                    "colorbar": {
                        "title": "请求数（热度）" + (" (log)" if log_color else "")
                    },
                },
                customdata=[[p[key] for key in custom_fields] for p in points],
                hovertemplate=hovertemplate,
                name=trace_name,
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "非缓存 compute tokens (X)",
            "yaxis_title": "缓存 reuse tokens (Y)",
            "zaxis_title": z_title,
            "zaxis_type": "log" if log_z else "linear",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.7},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
