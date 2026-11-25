"""Keep a rented run's results in an S3 bucket, so they don't depend on the
computer that launched it (which may be asleep, rebooting or gone when the
machine finishes and deletes itself).

    python3 train/run_storage.py push DIR NAME [--every 300]   upload changed files (every N s with --every)
    python3 train/run_storage.py pull NAME DIR                 download a run's files
    python3 train/run_storage.py get NAME/FILE DEST            download one file (e.g. a checkpoint to resume)
    python3 train/run_storage.py ls [NAME]                     list runs, or one run's files

Settings come from the environment: RUN_BUCKET, RUN_BUCKET_REGION and the
AWS key (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY), a key that can reach
nothing but this bucket's runs/ and datasets/. Files land under
runs/NAME/. A push uploads only files whose size or modification time
changed, and writes each file whole, so a reader never sees half of one.
"""
import argparse
import os
import sys
import time


def client():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("RUN_BUCKET_REGION", "us-east-1"))


def bucket():
    b = os.environ.get("RUN_BUCKET")
    if not b:
        raise SystemExit("RUN_BUCKET is not set")
    return b


def push_once(s3, root, name, seen):
    sent = 0
    for base, _, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            sig = (st.st_size, int(st.st_mtime))
            rel = os.path.relpath(p, root)
            if seen.get(rel) == sig:
                continue
            # A file still being written changes again before the next pass;
            # it is sent then.
            s3.upload_file(p, bucket(), f"runs/{name}/{rel}")
            seen[rel] = sig
            sent += 1
    return sent


def push(root, name, every):
    s3, seen = client(), {}
    while True:
        try:
            n = push_once(s3, root, name, seen)
            if n:
                print(f"[run_storage] {time.strftime('%H:%M:%S')} pushed {n} file(s) to s3://{bucket()}/runs/{name}/", flush=True)
        except Exception as error:  # noqa: BLE001 - keep trying; the next pass sends what this one missed
            print(f"[run_storage] push failed: {type(error).__name__}: {error}", flush=True)
            if not every:
                raise
        if not every:
            return
        time.sleep(every)


def listing(s3, prefix):
    out, token = [], None
    while True:
        kw = {"Bucket": bucket(), "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += r.get("Contents", [])
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("dir"); p.add_argument("name"); p.add_argument("--every", type=int, default=0)
    p = sub.add_parser("pull"); p.add_argument("name"); p.add_argument("dir")
    p = sub.add_parser("get"); p.add_argument("key"); p.add_argument("dest")
    p = sub.add_parser("ls"); p.add_argument("name", nargs="?")
    a = ap.parse_args()
    if a.cmd == "push":
        push(a.dir, a.name, a.every)
    elif a.cmd == "pull":
        s3 = client()
        for o in listing(s3, f"runs/{a.name}/"):
            dest = os.path.join(a.dir, o["Key"][len(f"runs/{a.name}/"):])
            if os.path.exists(dest) and os.path.getsize(dest) == o["Size"]:
                continue
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            s3.download_file(bucket(), o["Key"], dest)
            print(f"{o['Key']} -> {dest}")
    elif a.cmd == "get":
        client().download_file(bucket(), f"runs/{a.key}", a.dest)
    else:
        s3 = client()
        if a.name:
            for o in listing(s3, f"runs/{a.name}/"):
                print(f"{o['Size']:>13,}  {o['LastModified']:%Y-%m-%d %H:%M}  {o['Key'][len('runs/'):]}")
        else:
            names = sorted({o["Key"].split("/")[1] for o in listing(s3, "runs/") if o["Key"].count("/") >= 2})
            print("\n".join(names) or "(no runs yet)")


if __name__ == "__main__":
    sys.exit(main())
