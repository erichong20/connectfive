"""Native (C) guided MCTS core, loaded with ctypes; Python keeps the network.

``NativeMCTS`` has the same interface and result format as ``GuidedMCTS`` and
the same semantics as ``GuidedMCTS(batch_size=B)``: the C core picks up to B
leaves under virtual loss, Python evaluates them in one batched network call,
and the core expands and backs them up. Board, tactics and VCF run in C; VCF
results persist in the core across searches.

The shared library is compiled on first use with the system C compiler into
``_build/`` next to this file (ignored by Git).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sysconfig
import threading
import time
from pathlib import Path

import numpy as np

from connectfive.pattern_search import PatternSearchResult, PatternSearchStats

SOURCE = Path(__file__).with_name("engine.c")
ACTIONS = 225
_LIB = None
_LOCK = threading.Lock()


def _library() -> ctypes.CDLL:
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
        suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
        target = SOURCE.parent / "_build" / f"engine-{digest}{suffix}"
        if not target.exists():
            target.parent.mkdir(exist_ok=True)
            compiler = os.environ.get("CC", "cc")
            partial = target.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run([compiler, "-O3", "-std=c11", "-shared", "-fPIC", "-o", str(partial),
                            str(SOURCE), "-lm"], check=True)
            partial.replace(target)
        lib = ctypes.CDLL(str(target))
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_double_p = ctypes.POINTER(ctypes.c_double)
        lib.cf_new.restype = ctypes.c_void_p
        lib.cf_new.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib.cf_free.argtypes = [ctypes.c_void_p]
        lib.cf_clear_vcf_cache.argtypes = [ctypes.c_void_p]
        c_u64_p = ctypes.POINTER(ctypes.c_uint64)
        lib.cf_begin.argtypes = [ctypes.c_void_p, c_int_p, ctypes.c_int, c_float_p, c_u64_p]
        lib.cf_root_eval.argtypes = [ctypes.c_void_p, c_float_p, ctypes.c_float]
        lib.cf_root_children.argtypes = [ctypes.c_void_p, c_int_p, c_double_p]
        lib.cf_set_root_priors.argtypes = [ctypes.c_void_p, c_double_p]
        lib.cf_pick.argtypes = [ctypes.c_void_p, ctypes.c_int, c_float_p, c_u64_p, c_int_p]
        lib.cf_apply.argtypes = [ctypes.c_void_p, c_float_p, c_float_p]
        lib.cf_apply.restype = None
        lib.cf_result.argtypes = [ctypes.c_void_p, c_int_p, c_int_p, c_double_p, c_int_p]
        _LIB = lib
        return lib


def _ptr(array: np.ndarray, kind):
    return array.ctypes.data_as(ctypes.POINTER(kind))


class NativeCore:
    """One C search state (with its VCF cache); reuse it across moves and games."""

    def __init__(self, c_puct: float = 1.5, top_k: int = 16, leaf_vcf_depth: int = 4,
                 virtual_loss: float = 1.0, planes: int = 4):
        self.lib = _library()
        self.planes = planes
        self.handle = self.lib.cf_new(c_puct, top_k, leaf_vcf_depth, virtual_loss, planes)

    def __del__(self):
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.cf_free(handle)
            self.handle = None


class BatchEvaluator:
    """Batched network calls on a fixed batch size (one XLA compilation).

    Results are cached by position hash, like ``NetworkEvaluator``: consecutive
    searches in a game revisit many positions.
    """

    def __init__(self, params, config, batch_size: int):
        import jax
        import jax.numpy as jnp

        from connectfive.network import PolicyValueNetwork

        model = PolicyValueNetwork(config)
        self.config = config
        self.batch_size = batch_size
        self._apply = jax.jit(lambda x: model.apply(params, x))
        self._jnp = jnp
        self.calls = 0
        self.cache: dict[int, tuple[np.ndarray, float]] = {}

    @classmethod
    def from_evaluator(cls, evaluator, batch_size: int) -> BatchEvaluator:
        """Share parameters with a ``guided_search.NetworkEvaluator``."""

        return cls(evaluator.params, evaluator.config, batch_size)

    def __call__(self, features: np.ndarray, hashes=None) -> tuple[np.ndarray, np.ndarray]:
        count = len(features)
        logits = np.zeros((count, ACTIONS), dtype=np.float32)
        values = np.zeros(count, dtype=np.float32)
        missing = list(range(count))
        if hashes is not None:
            missing = []
            for i, key in enumerate(hashes):
                cached = self.cache.get(int(key))
                if cached is None:
                    missing.append(i)
                else:
                    logits[i], values[i] = cached
        if missing:
            padded = np.zeros((self.batch_size,) + features.shape[1:], dtype=np.float32)
            padded[:len(missing)] = features[missing]
            out_logits, out_values = self._apply(self._jnp.asarray(padded))
            self.calls += 1
            logits[missing] = np.asarray(out_logits[:len(missing)], dtype=np.float32)
            values[missing] = np.asarray(out_values[:len(missing)], dtype=np.float32)
            if hashes is not None:
                if len(self.cache) > 200_000:
                    self.cache.clear()
                for i in missing:
                    self.cache[int(hashes[i])] = (logits[i].copy(), float(values[i]))
        return logits, values


class NativeMCTS:
    """Drop-in for ``GuidedMCTS`` (batched semantics) backed by the C core."""

    def __init__(self, board, evaluator: BatchEvaluator, time_limit: float = 0.2,
                 max_simulations: int = 100_000, core: NativeCore | None = None,
                 root_noise: float = 0.0, noise_alpha: float = 0.3,
                 rng: np.random.Generator | None = None):
        self.moves = [int(action) for action in board.moves]
        self.evaluator = evaluator
        self.time_limit = time_limit
        self.max_simulations = max_simulations
        self.core = core or NativeCore(planes=evaluator.config.input_planes)
        self.root_noise = root_noise
        self.noise_alpha = noise_alpha
        self.rng = rng or np.random.default_rng(0)
        self.simulations = 0

    def run(self) -> PatternSearchResult:
        lib, handle, planes = self.core.lib, self.core.handle, self.core.planes
        batch = self.evaluator.batch_size
        started = time.perf_counter()
        deadline = started + self.time_limit
        moves = np.asarray(self.moves, dtype=np.int32)
        features = np.zeros((batch, 15, 15, planes), dtype=np.float32)
        hashes = np.zeros(batch, dtype=np.uint64)
        status = lib.cf_begin(handle, _ptr(moves, ctypes.c_int), len(moves), _ptr(features, ctypes.c_float),
                              _ptr(hashes, ctypes.c_uint64))
        if status == 0:
            logits, values = self.evaluator(features[:1], hashes[:1])
            status = lib.cf_root_eval(handle, _ptr(logits, ctypes.c_float), float(values[0]))
        actions = np.zeros(ACTIONS, dtype=np.int32)
        if status == 3 and self.root_noise:
            priors = np.zeros(ACTIONS, dtype=np.float64)
            count = lib.cf_root_children(handle, _ptr(actions, ctypes.c_int), _ptr(priors, ctypes.c_double))
            noise = self.rng.dirichlet([self.noise_alpha] * count)
            mixed = np.ascontiguousarray((1 - self.root_noise) * priors[:count] + self.root_noise * noise)
            lib.cf_set_root_priors(handle, _ptr(mixed, ctypes.c_double))
        reason = "forced" if status in (1, 2) else "mcts"
        done = ctypes.c_int(0)
        while status == 3 and self.simulations < self.max_simulations and time.perf_counter() < deadline:
            size = min(batch, self.max_simulations - self.simulations)
            pending = lib.cf_pick(handle, size, _ptr(features, ctypes.c_float),
                                  _ptr(hashes, ctypes.c_uint64), ctypes.byref(done))
            self.simulations += done.value
            if pending:
                logits, values = self.evaluator(features[:pending], hashes[:pending])
                lib.cf_apply(handle, _ptr(logits, ctypes.c_float), _ptr(values, ctypes.c_float))
                self.simulations += pending
        visits = np.zeros(ACTIONS, dtype=np.int32)
        root_q, simulations = ctypes.c_double(0.0), ctypes.c_int(0)
        count = lib.cf_result(handle, _ptr(actions, ctypes.c_int), _ptr(visits, ctypes.c_int),
                              ctypes.byref(root_q), ctypes.byref(simulations))
        children = list(zip(actions[:count].tolist(), visits[:count].tolist()))
        most = max(v for _, v in children)
        best = tuple(sorted(a for a, v in children if v == most))
        root_scores = tuple(sorted(children, key=lambda item: -item[1]))
        stats = PatternSearchStats(
            nodes=self.simulations, vcf_nodes=0, completed_depth=0, budget_exhausted=False,
            elapsed_ms=(time.perf_counter() - started) * 1_000, transposition_hits=0, cutoffs=0,
            reason=reason,
        )
        return PatternSearchResult(best, int(1000 * root_q.value), root_scores, stats)
