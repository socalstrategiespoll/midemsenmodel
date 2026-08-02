import os
import sys

import mi_senate_primary_model as m
import mi_civicapi_feed as civicapi
import mi_publish as publish

GIST_ID = os.environ.get("MI_GIST_ID")
GIST_TOKEN = os.environ.get("MI_GIST_TOKEN")


def main():
    if not GIST_ID or not GIST_TOKEN:
        print("Missing MI_GIST_ID or MI_GIST_TOKEN environment variables.")
        sys.exit(1)

    mi_model = m.MichiganSenateModel()

    try:
        updated = civicapi.update_model_from_civicapi(mi_model)
        print(f"[civicAPI] updated OK ({len(updated)}/83 counties)")
    except Exception as e:
        print("[civicAPI] FAILED:", e)

    try:
        snap = publish.publish_snapshot(mi_model, GIST_ID, GIST_TOKEN)
        margin = snap["statewide"]["point"]["margin"]
        n = snap["meta"]["observed_counties"]
        print(f"[Publish] OK — {n} counties in, El-Sayed margin {margin:+.1f}")
    except Exception as e:
        print("[Publish] FAILED:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
