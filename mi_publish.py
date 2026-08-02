"""
Publishes the Michigan Senate primary model's projection to a GitHub Gist,
using the model's own export_json() output plus a rolling history array
for the margin-over-time chart (same pattern as the AZ/SD publishers).
"""

import json
import requests

GIST_FILENAME = "mi_senate_primary_state.json"
MAX_HISTORY_POINTS = 500


def publish_snapshot(mi_model, gist_id, gist_token):
    projection = mi_model.get_projection()
    export_str = mi_model.export_json(projection)
    snapshot = json.loads(export_str)

    sim_margins = projection["statewide_ci"].get("simulations", [])
    if sim_margins:
        p_el_sayed = sum(1 for x in sim_margins if x > 0) / len(sim_margins) * 100
    else:
        p_el_sayed = 50.0
    snapshot["statewide"]["p_el_sayed_wins"] = round(p_el_sayed, 2)

    # Fetch existing history to append to (same rolling-history pattern as AZ/SD)
    headers = {
        "Authorization": f"token {gist_token}",
        "Accept": "application/vnd.github+json",
    }
    history = []
    try:
        resp = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=10)
        if resp.status_code == 200:
            files = resp.json().get("files", {})
            existing = files.get(GIST_FILENAME)
            if existing and existing.get("content"):
                old_data = json.loads(existing["content"])
                history = old_data.get("history", [])
    except Exception:
        pass  # fine to start fresh if this fails

    history.append({
        "timestamp": snapshot["timestamp"],
        "observed_counties": snapshot["meta"]["observed_counties"],
        "margin": snapshot["statewide"]["point"]["margin"],
    })
    history = history[-MAX_HISTORY_POINTS:]
    snapshot["history"] = history

    payload = {
        "files": {
            GIST_FILENAME: {
                "content": json.dumps(snapshot, indent=2)
            }
        }
    }
    resp = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers, json=payload, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Gist update failed ({resp.status_code}): {resp.text}")

    return snapshot
