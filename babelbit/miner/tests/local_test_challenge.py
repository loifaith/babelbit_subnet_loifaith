#!/usr/bin/env python3
"""
Local miner scoring test harness.

Reads challenge JSON files (from miner-test-data/), simulates the word-by-word
utterance revelation, calls a local miner's /predict endpoint, writes JSONL logs,
and scores them using the production scorer.

No Bittensor wallet, Chutes account, or live utterance engine required.

Usage:
    # Start the miner first (in another terminal):
    MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=distilgpt2 \
        uv run python babelbit/miner/serve_miner.py

    # Then run this script:
    uv run python babelbit/miner/tests/local_test_challenge.py \
        --challenge ../miner-test-data/en/npr/03/en-npr-003026.json

    # Or test a directory of challenge files:
    uv run python babelbit/miner/tests/local_test_challenge.py \
        --challenge ../miner-test-data/en/npr/03/ \
        --max-challenges 3 --max-dialogues 2
"""
import argparse
import asyncio
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


def load_challenge(path: Path) -> Dict[str, Any]:
    """Load a challenge JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_challenge_files(path: Path, max_challenges: Optional[int] = None) -> List[Path]:
    """Find challenge JSON files from a path (single file or directory)."""
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.json"))
        if max_challenges and len(files) > max_challenges:
            random.shuffle(files)
            files = files[:max_challenges]
            files.sort()
        return files
    print(f"Error: {path} is not a file or directory", file=sys.stderr)
    sys.exit(1)


def tokenize_utterance(utterance: str) -> List[str]:
    """Split an utterance into whitespace tokens."""
    return utterance.strip().split() if utterance.strip() else []


async def call_miner_predict(
    client: httpx.AsyncClient,
    miner_url: str,
    session_id: str,
    step: int,
    prefix: str,
    context: str,
    timeout: float = 30.0,
) -> str:
    """Call the miner /predict endpoint and return the prediction string."""
    payload = {
        "index": session_id,
        "step": step,
        "prefix": prefix,
        "context": context,
        "done": False,
        "prediction": "",
    }
    try:
        resp = await client.post(
            f"{miner_url}/predict",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            print(f"  [warn] Miner returned HTTP {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
            return ""
        data = resp.json()
        return data.get("prediction", "")
    except httpx.ConnectError:
        print(f"  [error] Cannot connect to miner at {miner_url}. Is it running?", file=sys.stderr)
        raise
    except httpx.ReadTimeout:
        print(f"  [warn] Miner prediction timed out after {timeout}s", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  [warn] Miner call failed: {e}", file=sys.stderr)
        return ""


async def run_dialogue(
    client: httpx.AsyncClient,
    miner_url: str,
    dialogue_uid: str,
    utterances: List[str],
    session_id: str,
    timeout: float,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Simulate the utterance engine flow for one dialogue.
    Returns (jsonl_path_placeholder, list_of_jsonl_lines) to be written by caller.
    """
    jsonl_lines: List[Dict[str, Any]] = []
    context_memory = ""

    for utt_index, utterance in enumerate(utterances):
        tokens = tokenize_utterance(utterance)
        if not tokens:
            continue

        ground_truth = utterance.strip()
        revealed: List[str] = []

        for step, token in enumerate(tokens):
            revealed.append(token)
            prefix = " ".join(revealed)

            prediction = await call_miner_predict(
                client=client,
                miner_url=miner_url,
                session_id=session_id,
                step=step,
                prefix=prefix,
                context=context_memory,
                timeout=timeout,
            )

            jsonl_lines.append({
                "event": "predicted",
                "utterance_index": utt_index,
                "step": step,
                "prediction": prediction,
            })

        jsonl_lines.append({
            "event": "utterance_complete",
            "utterance_index": utt_index,
            "ground_truth": ground_truth,
        })

        if context_memory:
            context_memory += f" EOF {ground_truth}"
        else:
            context_memory = ground_truth

    return dialogue_uid, jsonl_lines


def write_jsonl(lines: List[Dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def score_dialogue_log(jsonl_path: Path) -> Optional[Dict[str, Any]]:
    """Score a dialogue JSONL using the production scorer."""
    try:
        from babelbit.scoring.score_dialogue import score_jsonl
        return score_jsonl(jsonl_path, lex_weight=0.0, show_steps=False)
    except ImportError:
        print(
            "  [warn] Could not import production scorer (babelbit.scoring.score_dialogue).\n"
            "         Install the project first: uv sync",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"  [warn] Scoring failed for {jsonl_path.name}: {e}", file=sys.stderr)
        return None


async def check_miner_health(client: httpx.AsyncClient, miner_url: str) -> bool:
    """Check if the miner is reachable."""
    try:
        resp = await client.get(f"{miner_url}/healthz", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            model = data.get("model", "unknown")
            loaded = data.get("model_loaded", False)
            print(f"Miner healthy: model={model}, loaded={loaded}")
            return True
        print(f"Miner health check failed: HTTP {resp.status_code}", file=sys.stderr)
        return False
    except httpx.ConnectError:
        print(f"Cannot connect to miner at {miner_url}. Is it running?", file=sys.stderr)
        print(f"\nStart the miner first:", file=sys.stderr)
        print(f"  MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=distilgpt2 \\", file=sys.stderr)
        print(f"      uv run python babelbit/miner/serve_miner.py", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Miner health check error: {e}", file=sys.stderr)
        return False


async def run_challenge(
    challenge_path: Path,
    miner_url: str,
    output_dir: Path,
    max_dialogues: Optional[int] = None,
    timeout: float = 30.0,
    skip_scoring: bool = False,
) -> Optional[float]:
    """
    Run a full challenge file against the miner and return the challenge mean score.
    """
    challenge = load_challenge(challenge_path)
    challenge_uid = challenge.get("challenge_uid", challenge_path.stem)
    dialogues = challenge.get("dialogues", [])

    if not dialogues:
        print(f"  No dialogues found in {challenge_path.name}")
        return None

    if max_dialogues and len(dialogues) > max_dialogues:
        dialogues = dialogues[:max_dialogues]

    print(f"\nChallenge: {challenge_uid}  ({len(dialogues)} dialogue(s))")
    print("-" * 60)

    session_id = str(uuid.uuid4())
    dialogue_scores: List[float] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    async with httpx.AsyncClient() as client:
        for di, dlg in enumerate(dialogues):
            dialogue_uid = dlg.get("dialogue_uid", f"dlg_{di}")
            utterances = dlg.get("utterances", [])

            if not utterances:
                print(f"  Dialogue {dialogue_uid}: no utterances, skipping")
                continue

            print(f"  Dialogue {dialogue_uid}: {len(utterances)} utterances ... ", end="", flush=True)

            _, jsonl_lines = await run_dialogue(
                client=client,
                miner_url=miner_url,
                dialogue_uid=dialogue_uid,
                utterances=utterances,
                session_id=session_id,
                timeout=timeout,
            )

            log_name = f"dialogue_{challenge_uid}_{dialogue_uid}_{ts}.jsonl"
            log_path = output_dir / log_name
            write_jsonl(jsonl_lines, log_path)

            if skip_scoring:
                print(f"logged ({len(jsonl_lines)} events)")
                continue

            scored = score_dialogue_log(log_path)
            if scored:
                avg_u = scored.get("dialogue_summary", {}).get("average_U_best_early", 0.0)
                n_utt = len(scored.get("utterances", []))
                dialogue_scores.append(avg_u)
                print(f"avg U_best = {avg_u:.4f}  ({n_utt} scored utterances)")
            else:
                print("scoring unavailable")

    if not dialogue_scores:
        return None

    challenge_mean = sum(dialogue_scores) / len(dialogue_scores)
    print(f"\n  Challenge mean U: {challenge_mean:.4f}")
    return challenge_mean


async def async_main(args: argparse.Namespace) -> int:
    challenge_path = Path(args.challenge)
    miner_url = args.miner_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Babelbit Local Miner Scoring Test")
    print("=" * 60)
    print(f"Miner URL:   {miner_url}")
    print(f"Challenge:   {challenge_path}")
    print(f"Output dir:  {output_dir}")
    print()

    async with httpx.AsyncClient() as client:
        if not await check_miner_health(client, miner_url):
            return 1

    print()

    files = discover_challenge_files(challenge_path, max_challenges=args.max_challenges)
    if not files:
        print("No challenge files found.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} challenge file(s)")

    all_scores: List[float] = []
    for fpath in files:
        score = await run_challenge(
            challenge_path=fpath,
            miner_url=miner_url,
            output_dir=output_dir,
            max_dialogues=args.max_dialogues,
            timeout=args.timeout,
            skip_scoring=args.skip_scoring,
        )
        if score is not None:
            all_scores.append(score)

    print("\n" + "=" * 60)
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"Overall mean U across {len(all_scores)} challenge(s): {overall:.4f}")
    elif args.skip_scoring:
        print("Scoring was skipped. JSONL logs saved to:", output_dir)
    else:
        print("No scores produced. Check warnings above.")
    print("=" * 60)

    return 0


def main(
    challenge: str,
    miner_url: str = "http://localhost:8091",
    output_dir: str = "local_test_output",
    max_challenges: Optional[int] = None,
    max_dialogues: Optional[int] = None,
    timeout: float = 30.0,
    skip_scoring: bool = False,
) -> int:
    """Programmatic entry point (used by the bb CLI command)."""
    args = argparse.Namespace(
        challenge=challenge,
        miner_url=miner_url,
        output_dir=output_dir,
        max_challenges=max_challenges,
        max_dialogues=max_dialogues,
        timeout=timeout,
        skip_scoring=skip_scoring,
    )
    return asyncio.run(async_main(args))


def cli_main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Local miner scoring test: simulate challenges and score predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single challenge file
  %(prog)s --challenge ../miner-test-data/en/npr/03/en-npr-003026.json

  # Directory of challenges (picks 3 random files, 2 dialogues each)
  %(prog)s --challenge ../miner-test-data/en/npr/03/ --max-challenges 3 --max-dialogues 2

  # Skip scoring (just produce JSONL logs)
  %(prog)s --challenge ../miner-test-data/en/npr/03/en-npr-003026.json --skip-scoring
""",
    )
    ap.add_argument("--challenge", required=True, help="Path to a challenge JSON file or directory of them")
    ap.add_argument("--miner-url", default="http://localhost:8091", help="Miner predict endpoint base URL (default: http://localhost:8091)")
    ap.add_argument("--output-dir", default="local_test_output", help="Directory for JSONL logs and score files (default: local_test_output/)")
    ap.add_argument("--max-challenges", type=int, default=None, help="Max number of challenge files to process (random selection from directory)")
    ap.add_argument("--max-dialogues", type=int, default=None, help="Max dialogues per challenge (for quick tests)")
    ap.add_argument("--timeout", type=float, default=30.0, help="Prediction request timeout in seconds (default: 30)")
    ap.add_argument("--skip-scoring", action="store_true", help="Skip scoring, only produce JSONL logs")
    args = ap.parse_args()

    rc = asyncio.run(async_main(args))
    sys.exit(rc)


if __name__ == "__main__":
    cli_main()
