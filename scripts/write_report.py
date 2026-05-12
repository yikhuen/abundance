#!/usr/bin/env python3
"""Write agent workflow report for dashboard review."""

import json, sys
from datetime import datetime, timezone
from pathlib import Path

def update_report(stage: str, content: dict):
    """Update a workflow report stage."""
    path = Path("data/processed/workflow_status.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if path.exists():
        current = json.loads(path.read_text())
    current[stage] = content
    current["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(current, indent=2, default=str))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--content", default="")
    p.add_argument("--status", default="completed")
    args = p.parse_args()
    
    update_report(args.stage, {
        "title": args.title, "content": args.content,
        "status": args.status, "updated": datetime.now(timezone.utc).isoformat()
    })
    print(f"Report updated: {args.stage}")
