"""Render every numbered document from ground_truth.json. Never edit outputs by hand."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

REPO = Path(__file__).resolve().parents[2]
TEMPLATES = REPO / "interviewer" / "templates"
GT = json.loads((REPO / "interviewer" / "answer_keys" / "ground_truth.json").read_text())

OUTPUTS = {
    "model_card.md": REPO / "models" / "return-risk" / "MODEL_CARD.md",
    "risk-412.md": REPO / "interviewer" / "tickets" / "RISK-412.md",
    "part2-ticket.md": REPO / "interviewer" / "tickets" / "part2-serial-returners.md",
    "senior-answer-key.md": REPO / "interviewer" / "answer_keys" / "senior.md",
    "manager-answer-key.md": REPO / "interviewer" / "answer_keys" / "manager.md",
    "alex-pr-body.md": REPO / "interviewer" / "rendered" / "alex-pr-body.md",
}


def _context() -> dict[str, str]:
    ctx = {}
    for key, value in GT.items():
        if isinstance(value, float):
            ctx[key] = f"{value:.2f}"
            ctx[key + "_pct"] = f"{value:.0%}"
        else:
            ctx[key] = str(value)
    return ctx


def main() -> None:
    ctx = _context()
    for template_name, out_path in OUTPUTS.items():
        template = Template((TEMPLATES / template_name).read_text())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(template.substitute(ctx))
        print(f"rendered {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
