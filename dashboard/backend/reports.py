import logging
from pathlib import Path

from .storage import AnalyticsRepository


LOGGER = logging.getLogger(__name__)


class ReportService:
    def __init__(self, repository: AnalyticsRepository, reports_dir: Path):
        self.repository = repository
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def json_export(self, match_id: str = "default_match") -> Path:
        return self.repository.export_json(self.reports_dir / f"{match_id}_analytics.json", match_id)

    def csv_export(self, match_id: str = "default_match") -> Path:
        return self.repository.export_csv(self.reports_dir / f"{match_id}_player_history.csv", match_id)

    def pdf_report(self, match_id: str = "default_match") -> Path:
        output_path = self.reports_dir / f"{match_id}_match_report.pdf"
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            players = self.repository.list_players(match_id)
            with PdfPages(output_path) as pdf:
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                ax.set_title("Football Intelligence Match Report", fontsize=18, pad=20)
                lines = [
                    f"Match ID: {match_id}",
                    f"Players tracked: {len(players)}",
                    f"Total distance: {sum(float(p.get('distance_km') or 0) for p in players):.2f} km",
                    f"Total touches: {sum(int(p.get('ball_touches') or 0) for p in players)}",
                    "",
                    "Top activity:",
                ]
                top = sorted(players, key=lambda p: float(p.get("activity_score") or 0), reverse=True)[:8]
                lines.extend(
                    f"Player {p.get('player_id')} | Team {p.get('team')} | Activity {float(p.get('activity_score') or 0):.1f}"
                    for p in top
                )
                ax.text(0.05, 0.9, "\n".join(lines), va="top", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            return output_path
        except Exception as exc:
            LOGGER.error("PDF report generation failed: %s", exc)
            output_path.write_text("PDF generation failed. Check dashboard logs.", encoding="utf-8")
            return output_path
