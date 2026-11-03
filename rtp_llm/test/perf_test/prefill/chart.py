"""Offline prefill charts with optional embedded Plotly and portable SVGs."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


def _description(row: Mapping[str, Any]) -> str:
    return (
        f"Grid: {row['grid_name']}; case: {row['case_id']}; "
        f"Batch: {row['batch_size']}; source: {row['measurement_contract']}; "
        f"geometry: {json.dumps(row['request_distribution'], sort_keys=True)}"
    )


def _svg(rows, field, title):
    max_x = max(
        (sum(m["compute_length"] for m in r["request_distribution"]) for r in rows),
        default=1,
    )
    max_y = max((r[field] for r in rows), default=1)
    points = []
    for row in rows:
        x = (
            70
            + 670
            * sum(m["compute_length"] for m in row["request_distribution"])
            / max_x
        )
        y = 350 - 290 * row[field] / max_y
        points.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#2563eb">'
            f"<title>{html.escape(_description(row))}; {html.escape(title)}: {row[field]}</title></circle>"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420" role="img">'
        f'<title>{html.escape(title)}</title><rect width="800" height="420" fill="white"/>'
        f'<text x="70" y="25">{html.escape(title)}</text>'
        '<path d="M70 50 V350 H760" fill="none" stroke="black"/>'
        f'<text x="5" y="60">{max_y:.4g}</text><text x="45" y="355">0</text>'
        f'<text x="710" y="375">{max_x}</text>'
        '<text x="250" y="405">Total computed tokens per batch</text>'
        + "".join(points)
        + "</svg>"
    )


def render_charts(report: Mapping[str, Any]) -> dict[str, str]:
    """Return deterministic artifacts; no network, browser, or engine is needed."""
    artifacts = {}
    rows = [row for row in report["observations"] if row["fit_eligible"]]
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs
    except ImportError:
        go = None
    for name, field, title in (
        ("latency", "batch_latency_ms", "Batch latency (ms): median round maximum"),
        (
            "tpm",
            "tokens_per_minute",
            "Computed tokens per minute: measured round wall time",
        ),
    ):
        svg = _svg(rows, field, title)
        artifacts[f"{name}.svg"] = svg
        note = (
            f'<p>Report kind: {html.escape(report["report_kind"])}. '
            f'{len(report["excluded"])} invalid or missing cases excluded. '
            "Historical latency-derived rows are excluded from these charts.</p>"
            "<ul>"
            + "".join(
                f'<li>{html.escape(e["case_id"])}: {html.escape(e["reason"])}</li>'
                for e in report["excluded"]
            )
            + "</ul>"
        )
        batches = sorted({row["batch_size"] for row in rows})
        if go is None:
            body = (
                "<p>Batch: all (Plotly unavailable; hover points for geometry and source).</p>"
                + svg
            )
        else:
            figure = go.Figure()
            for batch in batches:
                members = [r for r in rows if r["batch_size"] == batch]
                figure.add_trace(
                    go.Scatter(
                        x=[
                            sum(m["compute_length"] for m in r["request_distribution"])
                            for r in members
                        ],
                        y=[r[field] for r in members],
                        mode="markers",
                        name=f"Batch {batch}",
                        text=[html.escape(_description(r)) for r in members],
                        hovertemplate="%{text}<br>Computed tokens: %{x}<br>Value: %{y}<extra></extra>",
                    )
                )
            buttons = [
                dict(
                    label="All batches",
                    method="update",
                    args=[{"visible": [True] * len(batches)}],
                )
            ]
            buttons.extend(
                dict(
                    label=f"Batch {batch}",
                    method="update",
                    args=[{"visible": [b == batch for b in batches]}],
                )
                for batch in batches
            )
            figure.update_layout(
                title=title,
                xaxis_title="Total computed tokens per batch",
                yaxis_title=title,
                updatemenus=[dict(buttons=buttons)],
            )
            # Encode data as JSON, escaping HTML delimiters so persisted names cannot close a script.
            payload = (
                figure.to_json()
                .replace("<", "\\u003c")
                .replace(">", "\\u003e")
                .replace("&", "\\u0026")
            )
            body = (
                '<div id="chart"></div><script>'
                + get_plotlyjs()
                + "</script><script>"
                + "const figure="
                + payload
                + ';Plotly.newPlot("chart",figure.data,figure.layout,{responsive:true});</script>'
            )
        artifacts[f"{name}.interactive.html"] = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(title)}</title></head><body><h1>{html.escape(title)}</h1>"
            + note
            + body
            + "</body></html>"
        )
    return artifacts
