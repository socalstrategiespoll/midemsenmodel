"""
Feed from civicAPI (civicapi.org) for the Michigan Senate Democratic primary
(El-Sayed vs Stevens vs McMorrow), race ID 84778, confirmed via the embed
code provided: https://www.civicapi.org/embed2/?p=...raceId":84778...

Uses the same domain and endpoint pattern already confirmed working for the
AZ and SD races tonight: https://civicapi.org/api/v2/race/<id>.

County names in civicAPI's response are expected as slugs (e.g. "st_clair"
for "St. Clair") based on the pattern observed in the SD race -- this is
normalized defensively below, but has not been directly verified against
this specific Michigan race's real response. Candidate name matching uses
surname substrings for the same reason -- robust to minor formatting
differences without needing to know the exact string in advance.
"""

import re
import requests

import mi_senate_primary_model as model

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RACE_ID = 84778
RACE_URL = f"https://civicapi.org/api/v2/race/{RACE_ID}"

CANDIDATE_TO_KEY = {
    "EL-SAYED": "el_sayed",
    "ELSAYED": "el_sayed",
    "STEVENS": "stevens",
    "MCMORROW": "mcmorrow",
    "MC MORROW": "mcmorrow",
}

# All counties the model knows about, for normalized-name matching
MODEL_COUNTIES = set(model.COUNTY_REGIONS.keys())


class CivicAPIError(Exception):
    pass


def fetch_race(race_id=RACE_ID, timeout=12):
    resp = requests.get(f"https://civicapi.org/api/v2/race/{race_id}", timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return resp.json()


def diagnose_structure(data):
    print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else type(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                print(f"  '{k}': list of {len(v)} items")
                if v:
                    print(f"    first item keys: {list(v[0].keys()) if isinstance(v[0], dict) else v[0]}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with keys {list(v.keys())}")
            else:
                print(f"  '{k}': {v!r}")


def _normalize_county_name(raw_name):
    """Match a raw county name/slug against the model's known county names,
    tolerant of underscores, periods, and casing differences."""
    cleaned = raw_name.replace("_", " ").replace(".", "").strip().lower()
    for known in MODEL_COUNTIES:
        known_cleaned = known.replace(".", "").strip().lower()
        if known_cleaned == cleaned:
            return known
    return None


def _candidate_key(name):
    name_upper = (name or "").upper().replace(".", "").replace("-", " ")
    name_upper_nospace = name_upper.replace(" ", "")
    for surname, key in CANDIDATE_TO_KEY.items():
        surname_clean = surname.replace(" ", "")
        if surname_clean in name_upper_nospace:
            return key
    return None


def find_county_breakdown(data):
    region_results = data.get("region_results")
    if not isinstance(region_results, dict):
        return None

    county_totals = {}
    for slug, entry in region_results.items():
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("name") or slug
        matched_name = _normalize_county_name(raw_name)
        if matched_name is None:
            continue

        totals = {"el_sayed": 0, "stevens": 0, "mcmorrow": 0}
        for c in entry.get("candidates", []):
            key = _candidate_key(c.get("name", ""))
            if key is not None:
                totals[key] += c.get("votes", 0) or 0
        county_totals[matched_name] = totals

    return county_totals if county_totals else None


def update_model_from_civicapi(mi_model):
    """mi_model: a MichiganSenateModel instance to feed results into."""
    data = fetch_race()
    county_breakdown = find_county_breakdown(data)

    if county_breakdown is None:
        raise CivicAPIError(
            "civicAPI response has no recognizable county-level breakdown — "
            "only a statewide total is available. Call diagnose_structure() "
            "on the raw response to inspect it."
        )

    updated = []
    for county_name, totals in county_breakdown.items():
        total_votes = totals["el_sayed"] + totals["stevens"] + totals["mcmorrow"]
        if total_votes == 0:
            continue  # civicAPI lists this county with placeholder zeros -- no real votes reported yet
        mi_model.add_results(county_name, totals["el_sayed"], totals["stevens"], totals["mcmorrow"])
        updated.append(county_name)

    return updated


if __name__ == "__main__":
    data = fetch_race()
    diagnose_structure(data)
    print()
    breakdown = find_county_breakdown(data)
    if breakdown:
        print(f"County breakdown found for {len(breakdown)} counties:")
        print(breakdown)
    else:
        print("No county breakdown found — statewide only.")
