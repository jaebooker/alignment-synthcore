"""Copied Go Concept Lab native transport/settings/rules helpers, unchanged.
Original files and source hashes are recorded in input-provenance.json.
Third-party native engines/models remain external pinned research dependencies.
"""
import hashlib,json,math,os,re,selectors,subprocess,time
from pathlib import Path
import numpy as np
SOURCE_COMMITS={'leela':'3f297889563bcbec671982c655996ccff63fa253','katago':'3c144b3e797185caa98e2328ae69757f36d791ba'}
GTP_COLUMNS='ABCDEFGHJKLMNOPQRSTUVWXYZ'
def _positive_integer(value,name,zero=False):
 if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<(0 if zero else 1):raise ValueError(name)
 return int(value)
class GTPClient:
    """Sequential GTP transport with bounded reads and retained stderr logs."""
    def __init__(self, command, *, log_path, env=None, timeout=60):
        self.timeout = timeout
        self.log_path = Path(log_path)
        self.log = open(log_path, "wb")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=env, bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.sequence = 0

    def request(self, command):
        if "\n" in command or "\r" in command:
            raise ValueError("one GTP command per request")
        self.sequence += 1
        self.process.stdin.write(f"{self.sequence} {command}\n".encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            self.buffer[:] = self.buffer.replace(b"\r\n", b"\n")
            separator = self.buffer.find(b"\n\n")
            if separator >= 0:
                response = bytes(self.buffer[:separator]).decode("utf-8").strip()
                del self.buffer[:separator + 2]
                if not response:
                    continue
                success = f"={self.sequence}"
                failure = f"?{self.sequence}"
                if response.startswith(failure) and (len(response) == len(failure) or response[len(failure)].isspace()):
                    raise RuntimeError(f"GTP rejected {command!r}: {response}")
                if not response.startswith(success) or (len(response) > len(success) and not response[len(success)].isspace()):
                    raise RuntimeError(f"unexpected GTP response: {response[:200]}")
                return response[len(success):].lstrip()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError(f"GTP timed out during {command!r}")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.log.flush()
                tail = self.log_path.read_bytes()[-4000:].decode("utf-8", errors="replace")
                raise RuntimeError(f"native engine exited during {command!r}: {tail}")
            self.buffer.extend(chunk)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.selector.close()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def gtp_to_canonical(vertex, size=19):
    """Return top-left [row,col], or None for pass; reject resign/invalid moves."""
    text = str(vertex).strip().upper()
    if text == "PASS":
        return None
    match = re.fullmatch(r"([A-HJ-Z])([1-9][0-9]*)", text)
    if not match:
        raise ValueError("native search returned invalid or resign move: " + text)
    col, board_row = GTP_COLUMNS.index(match.group(1)), int(match.group(2))
    if col >= size or board_row > size:
        raise ValueError("native move lies outside board")
    return [size - board_row, col]

def search_settings(engine_kind, visits, seed):
    """Stable settings contract; hash excludes ephemeral temporary paths."""
    engine_kind = {"leela-zero": "leela", "lz": "leela"}.get(engine_kind, engine_kind)
    if engine_kind not in SOURCE_COMMITS:
        raise ValueError("engine_kind must be leela or katago")
    visits, seed = _positive_integer(visits, "visits"), _positive_integer(seed, "seed", zero=True)
    shared = {"engine": engine_kind, "source_commit": SOURCE_COMMITS[engine_kind],
              "requested_root_visits": visits, "seed": seed, "search_threads": 1,
              "fresh_process_per_query": True, "pondering": False, "resignation": False,
              "policy_noise": False, "chosen_move_temperature": 0,
              "time_settings": "0 86400 1", "rules_override": "chinese" if engine_kind == "katago" else "Leela Zero built-in rules",
              "raw_before_search": False}
    if engine_kind == "leela":
        shared.update(symmetry="native seeded random symmetry per network evaluation; no identity-only CLI option",
                      arguments=["--gtp", "--threads", "1", "--visits", str(visits),
                                 "--seed", str(seed), "--noponder", "--resignpct", "0",
                                 "--randomcnt", "0", "--timemanage", "off"],
                      search_command="genmove", search_plays_move=True,
                      pass_handling="native Leela Zero pass heuristics retained; --dumbpass not enabled")
    else:
        cfg = {
            "logAllGTPCommunication": "false", "logSearchInfo": "true", "logToStderr": "true",
            "rules": "chinese", "numSearchThreads": 1, "numNNServerThreadsPerModel": 1,
            "nnMaxBatchSize": 1, "nnCacheSizePowerOfTwo": -1, "nnMutexPoolSizePowerOfTwo": 0,
            "maxVisits": visits, "maxPlayouts": visits, "maxTime": "1e20",
            "searchRandSeed": str(seed), "nnRandSeed": str(seed), "nnRandomize": "false",
            "nnForcedSymmetry": 0, "rootNumSymmetriesToSample": 1, "rootSymmetryPruning": "false",
            "rootNoiseEnabled": "false", "wideRootNoise": 0,
            "chosenMoveTemperature": 0, "chosenMoveTemperatureEarly": 0,
            "allowResignation": "false", "resignThreshold": -1.0, "ponderingEnabled": "false",
            "friendlyPass": "false", "cleanupBeforePass": "false",
            "searchFactorWhenWinning": 1.0, "playoutDoublingAdvantage": 0.0,
        }
        shared.update(symmetry="forced identity 0", config=cfg,
                      search_command="kata-search", search_plays_move=False,
                      pass_handling="friendlyPass and cleanupBeforePass disabled")
    return shared

def parse_search_log(engine_kind, text):
    """Extract actual visit totals; preserve complete diagnostics separately."""
    if engine_kind == "leela":
        matches = list(re.finditer(r"(?m)^(\d+) visits, (\d+) nodes, (\d+) playouts, ([0-9.eE+-]+) n/s", text))
        if not matches:
            raise ValueError("Leela Zero log has no completed visit count")
        last = matches[-1]
        result = {"root_visits": int(last[1]), "nodes": int(last[2]),
                  "new_playouts": int(last[3]), "reported_evaluations_per_second": float(last[4])}
        candidates = []
        pattern = r"(?m)^\s*(\S+) ->\s*(\d+) \(V:\s*([0-9.]+)%\) \(LCB:\s*([0-9.]+)%\) \(N:\s*([0-9.]+)%\) PV: (.*)$"
        for match in re.finditer(pattern, text[:last.start()]):
            candidates.append({"move": match[1], "visits": int(match[2]), "winrate_percent": float(match[3]),
                               "lcb_percent": float(match[4]), "prior_percent": float(match[5]), "pv": match[6].strip()})
        result["printed_candidates"] = candidates
        result["candidate_precision"] = "native diagnostics rounded to 2 decimal percentage points"
        return result
    if engine_kind == "katago":
        def last_number(label, cast=int):
            values = re.findall(re.escape(label) + r":\s*([0-9.eE+-]+)", text)
            if not values:
                raise ValueError("KataGo log has no " + label)
            return cast(values[-1])
        pvs = re.findall(r"(?m)^PV:\s*(.*)$", text)
        candidates = []
        pattern = r"(?m)^([A-HJ-Z][1-9][0-9]*|pss)\s*:\s+.*?\bP\s+([0-9.]+)%.*?\bN\s+(\d+)\s+--\s+(.*)$"
        for match in re.finditer(pattern, text):
            candidates.append({"move": "pass" if match[1] == "pss" else match[1],
                               "prior_percent": float(match[2]), "visits": int(match[3]),
                               "pv": match[4].strip()})
        return {"root_visits": last_number("Root visits"), "new_playouts": last_number("New playouts"),
                "nn_rows": last_number("NN rows"), "nn_batches": last_number("NN batches"),
                "engine_reported_seconds": last_number("Time taken", float),
                "pv": pvs[-1].strip() if pvs else None, "printed_candidates": candidates,
                "candidate_precision": "native top-10 tree diagnostics; priors rounded to 2 decimal percentage points"}
    raise ValueError("unknown engine kind")

def parse_native_board(text, size=19):
    """Read both native showboard formats, ignoring coordinates/last-move marks."""
    rows = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})(?!\d)(.*)$", line)
        if match is None or not 1 <= int(match[1]) <= size:
            continue
        cells = re.findall(r"[XO.+]", match[2])
        if len(cells) != size:
            continue
        row = size - int(match[1])
        if row in rows:
            raise ValueError("duplicate native board row")
        rows[row] = [1 if cell == "X" else -1 if cell == "O" else 0 for cell in cells]
    if set(rows) != set(range(size)):
        raise ValueError("native showboard did not contain one complete board")
    return np.asarray([rows[row] for row in range(size)], dtype=np.int8)

class _LineRulesState:
    """Independent sgfmill transitions, simple-ko, no-suicide and optional PSK."""
    def __init__(self, size):
        from sgfmill import boards
        self.size = size
        self.board = boards.Board(size)
        self.ko = None  # sgfmill bottom-left coordinates
        self.seen = {self.array().tobytes()}

    def array(self, board=None):
        b = self.board if board is None else board
        return np.asarray([[1 if b.get(self.size-1-y, x) == "b" else
                            -1 if b.get(self.size-1-y, x) == "w" else 0
                            for x in range(self.size)] for y in range(self.size)], dtype=np.int8)

    def propose(self, vertex, color, *, positional_superko):
        point = gtp_to_canonical(vertex, self.size)
        candidate = self.board.copy()
        if point is None:
            return candidate, None, self.array(candidate)
        row, col = self.size - 1 - point[0], point[1]
        if candidate.get(row, col) is not None:
            raise ValueError("occupied point")
        if (row, col) == self.ko:
            raise ValueError("simple-ko recapture")
        next_ko = candidate.play(row, col, color)
        if candidate.get(row, col) != color:
            raise ValueError("suicide")
        result = self.array(candidate)
        if positional_superko and result.tobytes() in self.seen:
            raise ValueError("positional repetition of a full-prefix board")
        return candidate, next_ko, result

    def commit(self, proposal):
        self.board, self.ko, array = proposal
        self.seen.add(array.tobytes())

def _katago_policy_allows(text, vertex, size):
    """KataGo's NN output marks illegal points as NAN, regardless of prior size."""
    lines = text.splitlines()
    try:
        start = lines.index("policy") + 1
        values = np.asarray([[float(v) for v in row.split()] for row in lines[start:start+size]])
        pass_value = float(next(row.split()[1] for row in lines if row.startswith("policyPass ")))
    except (ValueError, StopIteration) as error:
        raise ValueError("malformed native legality policy") from error
    if values.shape != (size, size):
        raise ValueError("native legality policy has wrong shape")
    point = gtp_to_canonical(vertex, size)
    value = pass_value if point is None else values[tuple(point)]
    return bool(np.isfinite(value) and value >= 0), None if not np.isfinite(value) else float(value)
