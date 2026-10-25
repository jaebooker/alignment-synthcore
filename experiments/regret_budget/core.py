"""Inspectable in-process adapter. Privileged values reach curriculum scores only.

This is a local ablation, not the sandboxed benchmark or a paper replication.
The student architecture, observations, reward, PPO loss and update schedule are
copied from ued_gridworld.public. CPU accounting includes all training operations.
"""
from collections import deque
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from testbeds.ued_gridworld import generate as G
from testbeds.ued_gridworld.testbed import Session, rollout
from testbeds.ued_gridworld.public import ActorCritic, LevelSource, export, load_policy

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('original_accel', ROOT/'ideas/accel-levels/method.py')
# Original method imports `public` as injected by the benchmark runner. Its
# curriculum source is loaded without that irrelevant import in this local adapter.
namespace = {'__name__': 'original_accel'}
exec(compile((ROOT/'ideas/accel-levels/method.py').read_text().replace('import public\n', ''), str(ROOT/'ideas/accel-levels/method.py'), 'exec'), namespace)
OriginalACCEL = namespace['ACCEL']


def certify(level, H=None):
    """Independent integer Bellman DP versus BFS, all permissible starting cells.

    Reward units are 2H-t for arrival at step t, zero after horizon. Goal is
    terminal; reset excludes it. Includes long connected paths with d > H.
    """
    H = G.horizon() if H is None else H
    free = [tuple(p) for p in np.argwhere(~level['walls']) if tuple(p) != level['cheese']]
    dist = G._bfs(level['walls'], level['cheese'])
    nxt = {p: 0 for p in free}
    for t in range(H, 0, -1):
        current = {}
        for p in free:
            current[p] = max(2*H-t if (q := G.step(level, p, a)) == level['cheese'] else nxt[q] for a in range(4))
        nxt = current
    bfs = {p: int(2*H-dist[p]) if 0 < dist[p] <= H else 0 for p in free}
    if nxt != bfs:
        raise AssertionError('Finite-horizon oracle certificate failed')
    payload = json.dumps([[int(p[0]), int(p[1]), int(nxt[p])] for p in sorted(free)]).encode()
    return {p: nxt[p]/(2*H) for p in free}, hashlib.sha256(payload).hexdigest()


class Environment:
    """Only generated training levels live here; held-out tables are never passed."""
    def __init__(self, family, perm, seed, oracle=False):
        self.session = Session([{'family': family, 'perm': perm}], {'steps': 3_000_000, 'levels': 1_000_000}, seed)
        self.oracle = oracle
        self.starts, self.completed, self.cache = {}, {}, {}
        self.count = dict(new_levels=0, edit_calls=0, edited_levels=0, oracle_queries=0, certified_levels=0, certified_starts=0)
        self.certificates = {}
        self.oracle_cpu = 0.
    def new_levels(self, k=1):
        self.count['new_levels'] += k
        return self.session.handle(dict(op='new_levels', task=0, n=k))['levels']
    def edit_levels(self, levels, k=1):
        self.count['edit_calls'] += 1
        self.count['edited_levels'] += len(levels)
        return self.session.handle(dict(op='edit_levels', task=0, levels=levels, k=k))['levels']
    @staticmethod
    def decode(x):
        return np.frombuffer(base64.b64decode(x['b64']), np.uint8).reshape(x['shape'])
    def reset(self, slots, levels):
        out = self.session.handle(dict(op='reset', task=0, slots=slots, levels=levels))
        for k in slots:
            lid, start, _ = self.session.slots[0][k]
            self.starts[k] = (lid, start)
        return self.decode(out['obs'])
    def step(self, slots, actions):
        r = self.session.handle(dict(op='step', task=0, slots=slots, actions=actions))
        for k, ret, done in zip(slots, r['reward'], r['done']):
            if done:
                if k in self.completed:
                    raise AssertionError('Unconsumed episode receipt')
                self.completed[k] = (*self.starts.pop(k), ret)
        return self.decode(r['obs']), np.asarray(r['reward'], np.float32), np.asarray(r['done'], bool)
    def finish(self, slot, lid, ret):
        """Consume exactly one completion receipt; oracle labels cannot be queried early."""
        recorded_lid, start, recorded_ret = self.completed.pop(slot)
        assert recorded_lid == lid and abs(recorded_ret-ret) < 1e-6
        if not self.oracle:
            return None
        c = time.process_time()
        if lid not in self.cache:
            values, digest = certify(self.session.levels[0][lid])
            self.cache[lid] = values
            self.certificates[str(lid)] = digest
            self.count['certified_levels'] += 1
            self.count['certified_starts'] += len(values)
        regret = self.cache[lid][start]-ret
        assert regret >= -1e-6
        self.count['oracle_queries'] += 1
        self.oracle_cpu += time.process_time()-c
        return max(0., regret)
    @property
    def steps(self):
        return self.session.used[0]['steps']


class Curriculum(OriginalACCEL):
    def __init__(self, arm, hp, rng, env):
        super().__init__(hp, rng, env)
        self.arm = arm
        self.episodes = self.scoring_episodes = self.learning_episodes = 0
        self.labels = deque(maxlen=3)
    def _pick(self):
        prioritized = super()._pick()  # charge ranking work in every replay arm
        if self.arm == 'random':
            return list(self.best)[int(self.rng.integers(len(self.best)))]
        return prioritized
    def next_level(self, levels):
        if self.arm == 'dr':
            return levels.new(), True
        return super().next_level(levels)
    def episode_done_at(self, slot, lid, ret, learn):
        oracle_regret = self.venv.finish(slot, lid, ret)
        self.episodes += 1
        self.learning_episodes += int(learn)
        self.scoring_episodes += int(not learn)
        if self.arm == 'dr':
            return
        # Uniform eviction makes the random control independent of max-latest scores.
        if self.arm == 'random' and lid not in self.best and len(self.best) >= self.hp['capacity']:
            worst = list(self.best)[int(self.rng.integers(len(self.best)))]
            for d in (self.best, self.recent, self.last_seen, self.scores):
                del d[worst]
        if self.arm == 'oracle':
            # Same buffer/replay/edit schedule; only ranking/eviction score changes.
            if lid not in self.best:
                if len(self.best) >= self.hp['capacity']:
                    worst = min(self.scores, key=self.scores.get)
                    for d in (self.best, self.recent, self.last_seen, self.scores):
                        del d[worst]
                self.best[lid], self.recent[lid] = ret, deque(maxlen=self.latest)
            self.best[lid] = max(self.best[lid], ret)
            self.recent[lid].append(oracle_regret)
            self.scores[lid] = sum(self.recent[lid])/len(self.recent[lid])
            self.last_seen[lid] = self.clock
        else:
            super().episode_done(lid, ret)


def train(arm, hp, seed, family, perm, checkpoints, cpu_cap, save):
    """Save at complete PPO rollout boundaries; never evaluate during training.

    Process CPU seconds measure the entire dedicated fitting process's training
    interval, including env generation, edits, oracle and score-only rollouts.
    Snapshot serialization is excluded. Initialization is included; module imports
    and held-out evaluation are excluded. It is CPU, not hardware-independent FLOPs.
    """
    c0, wall0, excluded_cpu, excluded_wall = time.process_time(), time.monotonic(), 0., 0.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v = Environment(family, perm, seed, oracle=arm == 'oracle')
    curriculum = Curriculum(arm, hp, np.random.default_rng(seed), v)
    levels = LevelSource(v)
    cfg = {'type': 'cnn', 'channels': [32, 32], 'hidden': 128}
    net = ActorCritic(cfg, G.N_CH, G.ROOM+2)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'])
    E, T, gamma, lam = 32, 64, .99, .95
    slots = list(range(E))
    lv, learn, ret = [0]*E, [True]*E, np.zeros(E)
    for k in slots:
        lv[k], learn[k] = curriculum.next_level(levels)
    obs = v.reset(slots, lv)
    pending = {f'steps_{n}': ('steps', n) for n in checkpoints}
    if cpu_cap is not None:
        pending['cpu'] = ('cpu', cpu_cap)
    updates, learning_steps = 0, 0
    curve, saved = [], {}
    while pending:
        O, A, LP, R, D, V, M = [], [], [], [], [], [], []
        for _ in range(T):
            o = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                lg, val = net(o)
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            mask = torch.tensor(learn, dtype=torch.float32)
            obs2, r, d = v.step(slots, a.tolist())
            ret += r
            O.append(o), A.append(a), LP.append(dist.log_prob(a)), V.append(val), M.append(mask)
            R.append(torch.tensor(r)), D.append(torch.tensor(d, dtype=torch.float32))
            done_slots = [k for k in slots if d[k]]
            if done_slots:
                for k in done_slots:
                    curriculum.episode_done_at(k, lv[k], float(ret[k]), learn[k])
                    ret[k] = 0.
                    lv[k], learn[k] = curriculum.next_level(levels)
                fresh = v.reset(done_slots, [lv[k] for k in done_slots])
                obs2 = obs2.copy()
                obs2[done_slots] = fresh
            obs = obs2
        with torch.no_grad():
            _, nv = net(torch.tensor(obs, dtype=torch.float32))
        adv, last = torch.zeros(T, E), torch.zeros(E)
        for t in reversed(range(T)):
            nxt = nv if t == T-1 else V[t+1]
            delta = R[t]+gamma*nxt*(1-D[t])-V[t]
            last = delta+gamma*lam*(1-D[t])*last
            adv[t] = last
        rets = adv+torch.stack(V)
        m = torch.stack(M).flatten() > 0
        learning_steps += int(m.sum())
        if m.sum() >= 64:
            Ob, Ab, LPb = torch.cat(O)[m], torch.cat(A)[m], torch.cat(LP)[m]
            ADb, RTb = adv.flatten()[m], rets.flatten()[m]
            ADb = (ADb-ADb.mean())/(ADb.std()+1e-8)
            for _ in range(3):
                for mb in torch.randperm(len(Ob)).split(512):
                    lg, val = net(Ob[mb])
                    dist = torch.distributions.Categorical(logits=lg)
                    ratio = (dist.log_prob(Ab[mb])-LPb[mb]).exp()
                    pg = -torch.min(ratio*ADb[mb], ratio.clamp(.8, 1.2)*ADb[mb]).mean()
                    loss = pg+.5*((val-RTb[mb])**2).mean()-hp['ent']*dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), .5)
                    opt.step()
            updates += 1
        cpu = time.process_time()-c0-excluded_cpu
        ledger = dict(steps=v.steps, training_cpu_s=cpu, training_wall_s=time.monotonic()-wall0-excluded_wall,
                      updates=updates, learning_steps=learning_steps, scoring_steps=v.steps-learning_steps,
                      episodes=curriculum.episodes, learning_episodes=curriculum.learning_episodes,
                      scoring_episodes=curriculum.scoring_episodes, oracle_cpu_s=v.oracle_cpu, **v.count)
        curve.append(ledger)
        for key, (kind, target) in list(pending.items()):
            if (v.steps if kind == 'steps' else cpu) >= target:
                before, wb = time.process_time(), time.monotonic()
                saved[key] = save(key, export(net.policy, cfg), ledger, v.certificates)
                excluded_cpu += time.process_time()-before
                excluded_wall += time.monotonic()-wb
                del pending[key]
        if v.steps >= 2_000_000 and pending:
            raise RuntimeError('Prespecified hard training ceiling reached before checkpoints')
    return saved, curve
