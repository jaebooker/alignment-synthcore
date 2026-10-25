"""Train small-MLP PPO policies with the vendored Bounded Agency trainer (unchanged)."""
import sys, json, time
import numpy as np
sys.path[:0] = ["src/experiments/preference_audit/vendor"]
import jax, jax.numpy as jnp
import ppo

def train(seeds, ent, arch="mlp_s", p=0.3, updates=300, chunk=50):
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    init, step = ppo.make_trainer(arch, chunk, ent)
    carry = init(keys, p); stats = []
    for i in range(updates // chunk):
        carry = step(carry); jax.block_until_ready(carry["params"])
        st = np.asarray(carry["stats"]); stats.append((st[:, 1] / np.maximum(st[:, 0], 1)).tolist())
    return carry["params"], stats

def save(path, params):
    np.savez_compressed(path, **{l + "__" + n: np.asarray(v) for l, r in params.items() for n, v in r.items()})

if __name__ == "__main__":
    out, ent = sys.argv[1], float(sys.argv[2]); seeds = [int(x) for x in sys.argv[3].split(",")]
    t0 = time.time(); params, stats = train(seeds, ent); save(out, params)
    print(json.dumps({"out": out, "entropy": ent, "seeds": seeds, "seconds": round(time.time() - t0, 1),
                      "gem_per_episode_final": stats[-1]}))
