"""Serve the comparison page on this machine and keep the flags made in it.

    python3 train/report_server.py DIR [--port 8796] [--open]

DIR holds the page (index.html, from analysis.py) and, beside it,
audio.json ({song key: audio file}) and flags.jsonl. Listens on 127.0.0.1
only: the page and the songs may be private.

    GET  /                 the page
    GET  /audio/KEY        a song's audio (byte ranges, so the player can seek)
    GET  /flags            every flag, oldest first
    POST /flags            add a flag: {song, difficulty, map, time, tags, note, notes}
    DELETE /flags/ID       remove a flag

A flag marks a moment in one map of one song as wrong ("awkward flow",
"unplayable", ...), with the map's notes around it, so it can become a rule
for the flow filter or a training example later.
"""
import argparse
import json
import os
import re
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 512 * 1024
FIELDS = {"song": str, "difficulty": str, "map": str, "time": (int, float), "tags": list, "note": str, "notes": list}


class Handler(BaseHTTPRequestHandler):
    root = "."

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def flags_path(self):
        return os.path.join(self.root, "flags.jsonl")

    def read_flags(self):
        if not os.path.exists(self.flags_path()):
            return []
        flags, gone = [], set()
        for line in open(self.flags_path()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("deleted"):
                gone.add(row["deleted"])
            else:
                flags.append(row)
        return [f for f in flags if f["id"] not in gone]

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send_file(os.path.join(self.root, "index.html"), "text/html; charset=utf-8")
        if self.path == "/flags":
            return self.send_json(self.read_flags())
        m = re.fullmatch(r"/audio/([A-Za-z0-9_.-]+)", self.path)
        if m:
            table = json.load(open(os.path.join(self.root, "audio.json"))) if os.path.exists(os.path.join(self.root, "audio.json")) else {}
            path = table.get(m.group(1))
            if path and os.path.isfile(path):
                kind = {"wav": "audio/wav", "ogg": "audio/ogg", "egg": "audio/ogg", "mp3": "audio/mpeg"}.get(path.rsplit(".", 1)[-1].lower(), "application/octet-stream")
                return self.send_file(path, kind)
        self.send_error(404)

    def send_file(self, path, kind):
        size = os.path.getsize(path)
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_POST(self):
        if self.path != "/flags":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > MAX_BODY:
            return self.send_error(413)
        try:
            flag = json.loads(self.rfile.read(n))
            for k, kind in FIELDS.items():
                if not isinstance(flag.get(k), kind):
                    raise ValueError(f"{k} missing or wrong type")
        except (ValueError, json.JSONDecodeError) as error:
            return self.send_json({"error": str(error)}, 400)
        flag = {k: flag[k] for k in FIELDS} | {"id": uuid.uuid4().hex[:12], "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(self.flags_path(), "a") as f:
            f.write(json.dumps(flag) + "\n")
        self.send_json(flag, 201)

    def do_DELETE(self):
        m = re.fullmatch(r"/flags/([0-9a-f]{12})", self.path)
        if not m:
            return self.send_error(404)
        with open(self.flags_path(), "a") as f:
            f.write(json.dumps({"deleted": m.group(1), "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        self.send_json({"deleted": m.group(1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    Handler.root = os.path.abspath(os.path.expanduser(a.dir))
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"serving {Handler.root} at {url} (this machine only); flags go to flags.jsonl there", flush=True)
    if a.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
