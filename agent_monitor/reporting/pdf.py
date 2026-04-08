from __future__ import annotations

from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate(df: pd.DataFrame, out: str = "data/reports/latest.pdf") -> str:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter

    y = height - 50
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "Agent Performance Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    total_rows = len(df)
    total_agents = int(df.agent_name.nunique()) if not df.empty and "agent_name" in df.columns else 0
    avg_h = float(df.hallucination_score.mean()) if not df.empty and "hallucination_score" in df.columns else 0.0
    avg_l = float(df.latency.mean()) if not df.empty and "latency" in df.columns else 0.0

    lines = [
        f"Total calls: {total_rows}",
        f"Agents: {total_agents}",
        f"Avg hallucination: {avg_h:.2f}",
        f"Avg latency: {avg_l:.2f}s",
    ]
    for line in lines:
        pdf.drawString(40, y, line)
        y -= 18

    if not df.empty and {"agent_name", "hallucination_score", "latency"}.issubset(df.columns):
        y -= 12
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Per-agent summary")
        y -= 18
        pdf.setFont("Helvetica", 10)

        summary = (
            df.groupby("agent_name", as_index=False)
            .agg(
                calls=("agent_name", "count"),
                avg_hallucination=("hallucination_score", "mean"),
                avg_latency=("latency", "mean"),
            )
            .sort_values("calls", ascending=False)
        )
        for _, row in summary.iterrows():
            text = (
                f"{row['agent_name']}: calls={int(row['calls'])}, "
                f"halluc={row['avg_hallucination']:.2f}, latency={row['avg_latency']:.2f}s"
            )
            pdf.drawString(40, y, text[:110])
            y -= 14
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)

    pdf.showPage()
    pdf.save()
    return str(out_path)
