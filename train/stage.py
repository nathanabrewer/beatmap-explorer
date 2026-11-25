"""Stage what a rented GPU gets: the training code and the BeatSaver list, nothing else.

    python3 -I train/stage.py ~/GPU-runs/stage/NAME [--manifest library/train/manifest.jsonl] [--secrets FILE.env]

Foreman uploads a whole folder, and this project's library/ holds files that
must stay on this machine (levels extracted from your own copy of the game,
covers, caches). So the upload is a fresh folder holding only train/*.py,
train/cloud_run.sh, train/requirements.txt and train/dataset/manifest.jsonl,
and this refuses to finish if anything else is there. --secrets copies an
env file (the run bucket and its key) in as train/run.env, mode 600, which
cloud_run.sh reads on the machine: the key then never passes through the
launcher's run settings or records.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    dest = os.path.abspath(os.path.expanduser(sys.argv[1]))
    manifest = os.path.join(ROOT, "library", "train", "manifest.jsonl")
    if "--manifest" in sys.argv:
        manifest = os.path.abspath(sys.argv[sys.argv.index("--manifest") + 1])
    if dest.startswith(ROOT + os.sep):
        raise SystemExit("Stage outside this project (it holds files that must not be uploaded).")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.join(dest, "train", "dataset"))
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py") or f in ("cloud_run.sh", "requirements.txt"):
            shutil.copy2(os.path.join(HERE, f), os.path.join(dest, "train", f))
    shutil.copy2(manifest, os.path.join(dest, "train", "dataset", "manifest.jsonl"))
    if "--secrets" in sys.argv:
        src = os.path.expanduser(sys.argv[sys.argv.index("--secrets") + 1])
        target = os.path.join(dest, "train", "run.env")
        shutil.copyfile(src, target)
        os.chmod(target, 0o600)
    allowed_ext = (".py", ".sh", ".txt", ".jsonl", "run.env")
    total = 0
    for base, _, files in os.walk(dest):
        for f in files:
            p = os.path.join(base, f)
            if not f.endswith(allowed_ext):
                raise SystemExit(f"unexpected file staged: {p}")
            total += os.path.getsize(p)
    lines = sum(1 for _ in open(os.path.join(dest, "train", "dataset", "manifest.jsonl")))
    print(f"staged {dest}: {total / 1e6:.1f} MB, {lines} maps in the list; nothing from library/ or the game")


if __name__ == "__main__":
    main()
