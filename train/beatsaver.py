"""Build the dataset recipe: which BeatSaver maps to learn from.

    python3 -I train/beatsaver.py [--out library/train/manifest.jsonl] [--delay 1.0]

Searches BeatSaver's public API (metadata only, no downloads) for three
tiers, newest first, 100 per page:
    curated      picked by BeatSaver's curators
    ranked       on the ScoreSaber or BeatLeader ranked leaderboards
    verified80   by verified mappers, rated 80 % or better
and keeps a map when:
    - nobody (uploader, BeatSaver's AI check or an admin) declared it AI-made;
    - it is under 15 minutes and over 30 seconds;
    - at least one difficulty is a standard characteristic and needs no
      Noodle Extensions or Mapping Extensions (those difficulties are dropped).
Writes one JSON line per map (id, hash, download URL, song, mapper, tiers,
rating, the kept difficulties with note jump speed and counts). The list of
ids and hashes is what can be shared; the maps themselves stay with whoever
downloads them. Resumable: maps already in the output are skipped.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.beatsaver.com/search/text/{page}?{query}"
USER_AGENT = "beatmap-explorer-dataset/0.1"
TIERS = {
    "curated": {"curated": "true"},
    "ranked": {"leaderboard": "Ranked"},
    "verified80": {"verified": "true", "minRating": "0.8"},
}
CHARACTERISTICS = {"Standard", "OneSaber", "NoArrows", "360Degree", "90Degree", "Legacy"}


def get(url, delay):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.load(r)
            time.sleep(delay)
            return body
        except urllib.error.HTTPError as e:
            wait = 30 if e.code == 429 else 5 * (attempt + 1)
            print(f"  HTTP {e.code}, waiting {wait} s", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"  {type(e).__name__}: {e}, retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"giving up on {url}")


def keep(doc):
    """The manifest record for a map, or None when it isn't used."""
    if doc.get("declaredAi", "None") != "None" or doc.get("automapper"):
        return None
    meta = doc.get("metadata", {})
    if not 30 <= (meta.get("duration") or 0) <= 900 or not meta.get("bpm"):
        return None
    versions = [v for v in doc.get("versions", []) if v.get("state") == "Published"] or doc.get("versions", [])[:1]
    if not versions:
        return None
    v = versions[0]
    diffs, dropped = [], 0
    for d in v.get("diffs", []):
        if d.get("characteristic") not in CHARACTERISTICS or d.get("ne") or d.get("me"):
            dropped += 1
            continue
        diffs.append({"char": d["characteristic"], "diff": d.get("difficulty"), "njs": d.get("njs"), "offset": d.get("offset"),
                      "notes": d.get("notes"), "bombs": d.get("bombs"), "walls": d.get("obstacles"), "nps": d.get("nps"),
                      "chroma": bool(d.get("chroma")), "events": d.get("events")})
    if not diffs:
        return None
    stats, uploader = doc.get("stats", {}), doc.get("uploader", {})
    return {"id": doc["id"], "hash": v.get("hash"), "url": v.get("downloadURL"), "name": doc.get("name", ""),
            "song": meta.get("songName", ""), "sub": meta.get("songSubName", ""), "artist": meta.get("songAuthorName", ""),
            "mapper": meta.get("levelAuthorName", ""), "uploader": uploader.get("name", ""), "verified": bool(uploader.get("verifiedMapper")),
            "bpm": meta.get("bpm"), "duration": meta.get("duration"), "score": stats.get("score"), "up": stats.get("upvotes"),
            "down": stats.get("downvotes"), "curated": bool(doc.get("curatedAt")), "rankedSS": bool(doc.get("ranked")),
            "rankedBL": bool(doc.get("blRanked")), "uploaded": doc.get("uploaded"), "tags": doc.get("tags", []),
            "diffs": diffs, "droppedDiffs": dropped, "tiers": []}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library", "train", "manifest.jsonl"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (be kind to BeatSaver)")
    ap.add_argument("--tiers", default=",".join(TIERS))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    maps = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            m = json.loads(line)
            maps[m["id"]] = m
    seen = {"total": 0, "kept": 0}
    for tier in a.tiers.split(","):
        page = 0
        while True:
            q = dict(TIERS[tier], sortOrder="Latest", pageSize="100")
            body = get(API.format(page=page, query=urllib.parse.urlencode(q)), a.delay)
            docs = body.get("docs", [])
            if not docs:
                break
            for doc in docs:
                seen["total"] += 1
                if doc["id"] in maps:
                    if tier not in maps[doc["id"]]["tiers"]:
                        maps[doc["id"]]["tiers"].append(tier)
                    continue
                rec = keep(doc)
                if rec:
                    rec["tiers"].append(tier)
                    maps[doc["id"]] = rec
                    seen["kept"] += 1
            print(f"{tier} page {page}: {len(docs)} maps, {len(maps)} kept so far", flush=True)
            page += 1
            if page % 10 == 0:
                _write(a.out, maps)
    _write(a.out, maps)
    diffs = sum(len(m["diffs"]) for m in maps.values())
    by_tier = {t: sum(1 for m in maps.values() if t in m["tiers"]) for t in TIERS}
    print(f"done: {len(maps)} maps, {diffs} difficulties; by tier {by_tier}; written to {a.out}")


def _write(path, maps):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for m in maps.values():
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
