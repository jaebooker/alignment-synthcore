"""Runs one method in a separate process.

Invoked as: python -I sandbox_exec.py <method.py> <sandbox_dir> <hparams.json> <seed> [rpc:<rfd>,<wfd>]

With rpc, the testbed is interactive: public.py talks to the harness over the two pipe
file descriptors, and the environment runs in the harness process.

The sandbox dir holds only public inputs (inputs.pkl) and the testbed's
public helper module (public.py). Ground truth never leaves the parent
process. v0 limitation: this is process isolation, not a filesystem jail;
a container is the planned next step.
"""
import importlib.util
import json
import pickle
import random
import sys
import traceback


def main():
    method_path, sandbox, hp_path, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, sandbox)
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
    except ImportError:
        pass

    if len(sys.argv) > 5 and sys.argv[5].startswith("rpc:"):
        rfd, wfd = (int(x) for x in sys.argv[5][4:].split(","))
        import public
        public._connect(rfd, wfd)

    with open(f"{sandbox}/inputs.pkl", "rb") as f:
        tasks = pickle.load(f)
    with open(hp_path) as f:
        hparams = json.load(f)

    spec = importlib.util.spec_from_file_location("method_under_test", method_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        outputs = mod.run(tasks, hparams, seed)
        if not isinstance(outputs, list) or len(outputs) != len(tasks):
            raise ValueError(f"run() must return a list of {len(tasks)} outputs")
        with open(f"{sandbox}/outputs.pkl", "wb") as f:
            pickle.dump(outputs, f)
    except Exception:
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
