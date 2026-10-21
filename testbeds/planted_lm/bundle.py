"""Package the GPU job, the baseline and the planted_lm ideas for a pod.

    python -m testbeds.planted_lm.bundle      -> results/_cache/planted_lm/bundle.tgz.b64

The pod's start command decodes it from the BUNDLE_B64 environment variable, so the
pod needs no repo access. Every idea in ideas/ with "testbed": "planted_lm" is included.
"""
import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "_cache" / "planted_lm"


def ideas():
    for d in sorted((ROOT / "ideas").iterdir()):
        f = d / "idea.json"
        if f.exists() and json.loads(f.read_text()).get("testbed") == "planted_lm":
            yield d


def build() -> str:
    meta = {"baseline": "baseline_topk_sae.py", "ideas": []}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(HERE / "job.py", "job.py")
        tar.add(HERE / "baseline_topk_sae.py", "baseline_topk_sae.py")
        for d in ideas():
            tar.add(d / "method.py", f"{d.name}.py")
            meta["ideas"].append({"id": d.name, "method": f"{d.name}.py"})
        blob = json.dumps(meta).encode()
        info = tarfile.TarInfo("bundle.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    return base64.b64encode(buf.getvalue()).decode()


# Pod start command: clone the public repo, extract the bundle, run. Logs carry the results.
START = ("git clone --depth 1 https://github.com/jaebooker/alignment-synthcore /repo && cd /repo && "
         "python -m testbeds.planted_lm.bundle --extract /job && "
         "pip install -q transformers==4.46.3 datasets==3.1.0 2>&1 | tail -n 2; "
         "python -u /job/job.py /job 2>&1; echo JOB_EXIT $?; sleep infinity")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--extract":
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(build())), mode="r:gz") as tar:
            tar.extractall(sys.argv[2], filter="data")
        print("bundle extracted to", sys.argv[2])
        raise SystemExit
    OUT.mkdir(parents=True, exist_ok=True)
    b = build()
    (OUT / "bundle.tgz.b64").write_text(b)
    print(f"{len(b)} base64 chars; ideas: {[d.name for d in ideas()]}")
    print("start command:", START)
