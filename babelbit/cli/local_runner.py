"""
Local runner: mirrors the mainnet validator-runner flow against local miners.

Reads challenge JSON files, simulates the utterance engine's token-by-token
revelation, calls each local miner's /predict endpoint at each step (same
schema as mainnet), and scores results through the production scoring pipeline.

Supports multiple miners -- the same challenge is sent to every miner
concurrently, just like mainnet.

No Bittensor wallet, Chutes account, utterance engine, or S3 required.

Usage (Terminal 1 -- miner A on port 8091):
    MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=distilgpt2 \
        MINER_AXON_PORT=8091 uv run python babelbit/miner/serve_miner.py

Usage (Terminal 2 -- miner B on port 8092):
    MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=gpt2 \
        MINER_AXON_PORT=8092 uv run python babelbit/miner/serve_miner.py

Usage (Terminal 3 -- local validator against both):
    uv run bb local-validate \
        --miner-url http://localhost:8091 \
        --miner-url http://localhost:8092
"""
import asyncio
import json
import logging
import random
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from babelbit.chute_template.schemas import BBPredictedUtterance
from babelbit.cli.runner import _score_miners_for_challenge, group_steps_into_utterances
from babelbit.utils.miner_registry import Miner
from babelbit.utils.validation_submission import ValidationSubmissionClient

logger = logging.getLogger("babelbit.local_runner")


# ---------------------------------------------------------------------------
# Default test-data search paths (relative to the project root)
# ---------------------------------------------------------------------------

_TEST_DATA_CANDIDATES = [
    "../miner-test-data",
    "miner-test-data",
    "../miner-test-data/en/npr",
]


def _find_test_data_dir() -> Optional[Path]:
    """Locate the miner-test-data directory automatically."""
    for candidate in _TEST_DATA_CANDIDATES:
        p = Path(candidate).resolve()
        if p.is_dir() and any(p.rglob("*.json")):
            return p
    return None


# ---------------------------------------------------------------------------
# Challenge I/O
# ---------------------------------------------------------------------------

def _load_challenge(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_challenge_files(
    path: Path, max_challenges: Optional[int] = None
) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        if not files:
            return []
        if max_challenges and len(files) > max_challenges:
            files = random.sample(files, max_challenges)
        return files
    logger.error("%s is not a valid file or directory", path)
    return []


def _tokenize(text: str) -> List[str]:
    return text.strip().split() if text.strip() else []


# ---------------------------------------------------------------------------
# Synthetic miner creation
# ---------------------------------------------------------------------------

def _make_local_miner(
    miner_url: str,
    uid: int = 0,
    hotkey: Optional[str] = None,
) -> Miner:
    """Build a Miner dataclass pointing at a local HTTP endpoint."""
    from urllib.parse import urlparse
    parsed = urlparse(miner_url)
    return Miner(
        uid=uid,
        hotkey=hotkey or f"local_miner_{uid}_{uuid.uuid4().hex[:8]}",
        model="local",
        revision=None,
        slug=None,
        chute_id=None,
        block=0,
        axon_ip=parsed.hostname or "127.0.0.1",
        axon_port=parsed.port or 8091,
    )


# ---------------------------------------------------------------------------
# Miner prediction (direct HTTP, no Bittensor headers)
# ---------------------------------------------------------------------------

async def _call_local_predict(
    client: httpx.AsyncClient,
    miner_url: str,
    payload: BBPredictedUtterance,
    timeout: float,
) -> str:
    body = {
        "index": payload.index,
        "step": payload.step,
        "prefix": payload.prefix,
        "context": payload.context,
        "done": payload.done,
        "prediction": "",
    }
    try:
        resp = await client.post(
            f"{miner_url}/predict", json=body, timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning("Miner returned HTTP %s", resp.status_code)
            return ""
        return resp.json().get("prediction", "")
    except httpx.ConnectError:
        logger.error("Cannot connect to miner at %s", miner_url)
        raise
    except httpx.ReadTimeout:
        logger.warning("Miner prediction timed out")
        return ""
    except Exception as exc:
        logger.warning("Miner call error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Simulate utterance engine: step through challenge data token-by-token
# and query every local miner at every step
# (mirrors predict_with_utterance_engine_multi_miner)
# ---------------------------------------------------------------------------

async def _predict_one_miner(
    client: httpx.AsyncClient,
    miner_url: str,
    payload: BBPredictedUtterance,
    timeout: float,
) -> Tuple[str, str]:
    """Call a single miner and return (miner_url, prediction)."""
    prediction = await _call_local_predict(client, miner_url, payload, timeout)
    return miner_url, prediction


async def _simulate_challenge(
    client: httpx.AsyncClient,
    miners: List[Tuple[str, Miner]],
    challenge: Dict[str, Any],
    max_dialogues: Optional[int],
    timeout: float,
) -> Tuple[str, Dict[str, Dict[str, List[BBPredictedUtterance]]]]:
    """
    Walk every dialogue / utterance / token exactly like the mainnet runner,
    query ALL miners concurrently at each step, and return the populated
    miner_dialogues dict keyed by each miner's hotkey.
    """
    challenge_uid = challenge.get("challenge_uid", str(uuid.uuid4()))
    dialogues = challenge.get("dialogues", [])
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    session_id = str(uuid.uuid4())

    url_to_miner = {url: m for url, m in miners}

    miner_dialogues: Dict[str, Dict[str, List[BBPredictedUtterance]]] = {
        m.hotkey: {} for _, m in miners
    }
    miner_contexts: Dict[str, str] = {}

    for dlg in dialogues:
        dialogue_uid = dlg.get("dialogue_uid", str(uuid.uuid4()))
        utterances: List[str] = dlg.get("utterances", [])
        if not utterances:
            continue

        for _, m in miners:
            miner_dialogues[m.hotkey][dialogue_uid] = []
        miner_contexts[dialogue_uid] = ""

        logger.info(
            "  Dialogue %s: %d utterances", dialogue_uid[:12], len(utterances)
        )

        for utt_index, utterance in enumerate(utterances):
            tokens = _tokenize(utterance)
            if not tokens:
                continue

            ground_truth = utterance.strip()
            revealed: List[str] = []

            for step, token in enumerate(tokens):
                revealed.append(token)
                prefix = " ".join(revealed)
                context = miner_contexts.get(dialogue_uid, "")

                payload = BBPredictedUtterance(
                    index=session_id,
                    step=step,
                    prefix=prefix,
                    prediction="",
                    context=context,
                    done=False,
                )

                tasks = [
                    _predict_one_miner(client, url, payload, timeout)
                    for url in url_to_miner
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, httpx.ConnectError):
                        raise res
                    if isinstance(res, Exception):
                        logger.warning("Miner call failed: %s", res)
                        continue
                    miner_url_r, prediction = res
                    miner_key = url_to_miner[miner_url_r].hotkey
                    step_utt = BBPredictedUtterance(
                        index=session_id,
                        step=step,
                        prefix=prefix,
                        prediction=prediction,
                        context=context,
                        done=False,
                    )
                    miner_dialogues[miner_key][dialogue_uid].append(step_utt)

            for _, m in miners:
                steps = miner_dialogues[m.hotkey].get(dialogue_uid, [])
                if steps:
                    steps[-1].done = True
                    steps[-1].ground_truth = ground_truth

            existing = miner_contexts.get(dialogue_uid, "")
            miner_contexts[dialogue_uid] = (
                f"{existing} EOF {ground_truth}" if existing else ground_truth
            )

    return challenge_uid, miner_dialogues


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def _check_miner_health(client: httpx.AsyncClient, miner_url: str) -> bool:
    try:
        resp = await client.get(f"{miner_url}/healthz", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(
                "Miner healthy: model=%s  loaded=%s",
                data.get("model", "?"),
                data.get("model_loaded", False),
            )
            return True
        logger.error("Miner health check returned HTTP %s", resp.status_code)
        return False
    except httpx.ConnectError:
        logger.error(
            "Cannot reach miner at %s. Start the miner first:\n"
            "  MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=distilgpt2 "
            "uv run python babelbit/miner/serve_miner.py",
            miner_url,
        )
        return False
    except Exception as exc:
        logger.error("Miner health check error: %s", exc)
        return False


async def _check_all_miners_health(
    client: httpx.AsyncClient, miner_urls: List[str],
) -> List[str]:
    """Health-check every miner URL. Return only the healthy ones."""
    healthy: List[str] = []
    for url in miner_urls:
        if await _check_miner_health(client, url):
            healthy.append(url)
    return healthy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def local_validate(
    challenge: Optional[str] = None,
    miner_urls: Optional[List[str]] = None,
    output_dir: str = "local_test_output",
    max_challenges: Optional[int] = None,
    max_dialogues: Optional[int] = None,
    timeout: float = 30.0,
) -> int:
    """
    Run the full local validation loop: read challenges, call every miner at
    every token step, score with the production scorer, and print results.

    When *challenge* is None the runner automatically picks random file(s)
    from the ``miner-test-data/`` directory (defaults to 1 challenge).

    *miner_urls* is a list of miner base URLs.  Defaults to
    ``["http://localhost:8091"]`` when not provided.

    Returns 0 on success, 1 on error.
    """
    if not miner_urls:
        miner_urls = ["http://localhost:8091"]
    miner_urls = [u.rstrip("/") for u in miner_urls]

    # Resolve challenge path -- auto-discover test data when not specified
    if challenge is None:
        test_dir = _find_test_data_dir()
        if test_dir is None:
            print(
                "Error: No --challenge given and could not locate miner-test-data/ directory.\n"
                "  Provide --challenge explicitly or run from the project root.",
                file=sys.stderr,
            )
            return 1
        challenge_path = test_dir
        if max_challenges is None:
            max_challenges = 1
        print(f"Auto-selected test data directory: {challenge_path}")
    else:
        challenge_path = Path(challenge)

    logs_dir = Path(output_dir) / "logs"
    scores_dir = Path(output_dir) / "scores"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Babelbit Local Validator (mainnet-equivalent flow)")
    print("=" * 64)
    print(f"  Miners ({len(miner_urls)}):")
    for i, u in enumerate(miner_urls):
        print(f"    [{i}] {u}")
    print(f"  Challenge:    {challenge_path}")
    print(f"  Output dir:   {output_dir}")
    print()

    # Health-check all miners, keep only healthy ones
    async with httpx.AsyncClient() as client:
        healthy_urls = await _check_all_miners_health(client, miner_urls)
    if not healthy_urls:
        print("Error: No healthy miners found. Aborting.", file=sys.stderr)
        return 1
    if len(healthy_urls) < len(miner_urls):
        skipped = set(miner_urls) - set(healthy_urls)
        print(f"  Warning: skipping unhealthy miners: {', '.join(skipped)}\n")

    files = _discover_challenge_files(challenge_path, max_challenges)
    if not files:
        logger.error("No challenge files found at %s", challenge_path)
        return 1

    print(f"Found {len(files)} challenge file(s)\n")

    # Build (url, Miner) pairs with sequential UIDs
    miners: List[Tuple[str, Miner]] = [
        (url, _make_local_miner(url, uid=i))
        for i, url in enumerate(healthy_urls)
    ]
    miner_list = [m for _, m in miners]

    submission_client = ValidationSubmissionClient(enabled=False)

    # Per-miner score accumulators across all challenges
    miner_scores_all: Dict[str, List[float]] = {m.hotkey: [] for m in miner_list}

    async with httpx.AsyncClient() as client:
        for fpath in files:
            data = _load_challenge(fpath)
            n_dlg = len(data.get("dialogues", []))
            cuid_preview = data.get("challenge_uid", fpath.stem)[:16]
            print(f"Challenge {cuid_preview}…  ({n_dlg} dialogue(s), {len(miners)} miner(s))")
            print("-" * 64)

            try:
                challenge_uid, miner_dialogues = await _simulate_challenge(
                    client=client,
                    miners=miners,
                    challenge=data,
                    max_dialogues=max_dialogues,
                    timeout=timeout,
                )
            except httpx.ConnectError:
                return 1

            total_steps = sum(
                len(steps)
                for dlg_steps in miner_dialogues.values()
                for steps in dlg_steps.values()
            )
            logger.info(
                "Challenge %s: collected %d prediction steps across %d miner(s)",
                challenge_uid, total_steps, len(miners),
            )

            # --- Use the REAL production scoring pipeline from runner.py ---
            miners_processed, dialogues_processed, scores = (
                await _score_miners_for_challenge(
                    challenge_uid=challenge_uid,
                    challenge_type="local",
                    miner_list=miner_list,
                    miner_dialogues=miner_dialogues,
                    logs_dir=logs_dir,
                    scores_dir=scores_dir,
                    submission_client=submission_client,
                    active_s3_manager=None,
                )
            )

            if scores:
                print(
                    f"\n  Scored {dialogues_processed} dialogue(s) "
                    f"for {miners_processed} miner(s)"
                )
                for idx, (url, m) in enumerate(miners):
                    if idx < len(scores):
                        miner_scores_all[m.hotkey].append(scores[idx])
                        print(f"    Miner [{idx}] {url}: U = {scores[idx]:.4f}")
            else:
                print("  (no scores produced)")
            print()

    # Final summary
    print("=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    any_scores = False
    for idx, (url, m) in enumerate(miners):
        s = miner_scores_all[m.hotkey]
        if s:
            any_scores = True
            mean = sum(s) / len(s)
            print(f"  Miner [{idx}] {url}")
            print(f"    Mean U across {len(s)} challenge(s): {mean:.4f}")
    if not any_scores:
        print("  No scores produced. Check warnings above.")
    print("=" * 64)
    return 0


def main(
    challenge: Optional[str] = None,
    miner_urls: Optional[List[str]] = None,
    output_dir: str = "local_test_output",
    max_challenges: Optional[int] = None,
    max_dialogues: Optional[int] = None,
    timeout: float = 30.0,
) -> int:
    """Synchronous wrapper for the CLI."""
    return asyncio.run(
        local_validate(
            challenge=challenge,
            miner_urls=miner_urls,
            output_dir=output_dir,
            max_challenges=max_challenges,
            max_dialogues=max_dialogues,
            timeout=timeout,
        )
    )
