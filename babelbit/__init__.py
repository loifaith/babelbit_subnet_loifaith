import click
import sys
from asyncio import run
from pathlib import Path
from logging import getLogger, DEBUG, INFO, WARNING, basicConfig
import asyncio

from babelbit.cli.runner import runner_loop
from babelbit.cli.push import push_ml_model
from babelbit.utils.settings import get_settings
from babelbit.utils.bittensor_helpers import test_metagraph
from babelbit.cli.signer_api import run_signer
from babelbit.cli.validate import _validate_main
from babelbit.utils.prometheus import _start_metrics
from babelbit.chute_template.test import (
    deploy_mock_chute,
    test_chute_health_endpoint,
    test_chute_predict_endpoint,
    get_chute_logs,
    create_test_utterances,
)
from babelbit.utils.chutes_helpers import (
    render_chute_template,
    get_chute_slug_and_id,
    delete_chute,
)
# from babelbit.cli.run_vlm_pipeline import vlm_pipeline

logger = getLogger(__name__)


@click.group(name="bb")
@click.option(
    "-v",
    "--verbosity",
    count=True,
    help="Increase verbosity (-v INFO, -vv DEBUG)",
)
def cli(verbosity: int):
    """Babelbit CLI"""
    settings = get_settings()
    basicConfig(
        level=DEBUG if verbosity == 2 else INFO if verbosity == 1 else WARNING,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.debug(f"Babelbit started (version={settings.BABELBIT_VERSION})")


@cli.command("runner")
def runner_cmd():
    """Launches runner every TEMPO blocks."""
    asyncio.run(runner_loop())


@cli.command("push")
@click.option(
    "--model-path",
    help="Local path to model artifacts. If none provided, upload skipped",
)
@click.option(
    "--revision",
    default=None,
    help="Explicit revision SHA to commit (otherwise auto-detected).",
)
@click.option("--no-deploy", is_flag=True, help="Skip Chutes deployment (HF only).")
@click.option(
    "--no-commit", is_flag=True, help="Skip on-chain commitment (print payload only)."
)
@click.option(
    "--no-warmup", is_flag=True, help="Skip chute warmup (deploy only)."
)
def push(
    model_path,
    revision,
    no_deploy,
    no_commit,
    no_warmup,
):
    try:
        run(
            push_ml_model(
                ml_model_path=Path(model_path) if model_path else None,
                hf_revision=revision,
                skip_chutes_deploy=no_deploy,
                skip_bittensor_commit=no_commit,
                skip_warmup=no_warmup,
            )
        )
    except Exception as e:
        click.echo(e)


@cli.command("signer")
def signer_cmd():
    asyncio.run(run_signer())


@cli.command("validate")
@click.option(
    "--tail", type=int, envvar="BABELBIT_TAIL", default=28800, show_default=True
)
@click.option(
    "--alpha", type=float, envvar="BABELBIT_ALPHA", default=0.2, show_default=True
)
@click.option(
    "--m-min", type=int, envvar="BABELBIT_M_MIN", default=25, show_default=True
)
@click.option(
    "--tempo", type=int, envvar="BABELBIT_TEMPO", default=100, show_default=True
)
def validate_cmd(tail: int, alpha: float, m_min: int, tempo: int):
    """
    Babelbit validator (mainnet cadence):
      - attend block%tempo==0
      - calcule (uids, weights) winner-takes-all
      - push via signer, fallback local si signer HS
    """
    _start_metrics()
    # Overriding to keep validators in sync
    tempo = 100
    asyncio.run(_validate_main(tail=tail, alpha=alpha, m_min=m_min, tempo=tempo))


@cli.command("deploy-local-chute")
@click.option("--repo", type=str, default="distilgpt2", required=True)
@click.option(
    "--revision",
    type=str,
    default="main",
    required=True,
)
def deploy_chute_locally(repo: str, revision: str):
    """Locally deploy a chute for utterance prediction testing on localhost:8000 (uses OpenAI API)"""
    deploy_mock_chute(
        huggingface_repo=repo,
        huggingface_revision=revision,
    )


@cli.command("ping-chute")
@click.option("--local", is_flag=True, help="Use locally deployed mock chute server.")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def test_chute(revision: str, local: bool) -> None:
    """Check the response of the model endpoints for utterance prediction"""
    if local:
        base_url = "http://localhost:8000"
    else:
        slug, _ = run(get_chute_slug_and_id(revision=revision))
        settings = get_settings()
        base_url = settings.CHUTES_MINER_BASE_URL_TEMPLATE.format(
            slug=slug,
        )
    
    run(test_chute_health_endpoint(base_url=base_url))
    run(
        test_chute_predict_endpoint(
            base_url=base_url, test_utterances=create_test_utterances()
        )
    )


@cli.command("chute-slug")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def query_chute_slug(revision: str) -> None:
    chute_slug, chute_id = run(get_chute_slug_and_id(revision=revision))
    click.echo(f"Slug: {chute_slug}\nID: {chute_id}")


@cli.command("delete-chute")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def delete_model_from_chutes(revision: str) -> None:
    try:
        run(delete_chute(revision=revision))
    except Exception as e:
        click.echo(e)


@cli.command("chute-logs")
@click.option("--instance-id", type=str, required=True)
def chute_logs(instance_id: str) -> None:
    try:
        run(get_chute_logs(instance_id=instance_id))
    except Exception as e:
        click.echo(e)


@cli.command("generate-chute-script")
@click.option(
    "--revision",
    type=str,
    required=True,
)
def generate_chute_file(revision: str) -> None:
    with open("my_chute.py", "w+") as f:
        template = render_chute_template(
            revision=revision,
        )
        f.write(str(template))
        f.flush()


# @cli.command("test-chute")
# @click.option(
#     "--revision",
#     type=str,
#     required=True,
# )
# @click.option(
#     "--local", is_flag=True, help="Use locally deployed mock chute server for model"
# )
# def test_vlm_pipeline(revision: str, local: bool) -> None:
#     """Run the miner on the VLM-as-Judge pipeline off-chain (results not saved)"""
#     try:
#         result = run(vlm_pipeline(hf_revision=revision, local_model=local))
#         click.echo(result)
#     except Exception as e:
#         click.echo(e)


@cli.command("local-validate")
@click.option(
    "--challenge",
    default=None,
    help="Path to a challenge JSON file or directory. "
    "If omitted, a random challenge is picked from miner-test-data/.",
)
@click.option(
    "--miner-url",
    multiple=True,
    help="Miner URL (repeatable). "
    "Example: --miner-url http://localhost:8091 --miner-url http://localhost:8092. "
    "Defaults to http://localhost:8091 when none given.",
)
@click.option(
    "--output-dir",
    default="local_test_output",
    show_default=True,
    help="Directory for logs and score output.",
)
@click.option(
    "--max-challenges",
    type=int,
    default=None,
    help="Max challenge files to process (default: 1 when auto-selecting).",
)
@click.option(
    "--max-dialogues",
    type=int,
    default=None,
    help="Max dialogues per challenge.",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Prediction request timeout in seconds.",
)
def local_validate_cmd(challenge, miner_url, output_dir, max_challenges, max_dialogues, timeout):
    """Run the full local validator flow against local miners (mainnet-equivalent).

    Supports multiple miners -- pass --miner-url once per miner.  The same
    challenge is sent to every miner concurrently, just like mainnet.

    When --challenge is omitted, a random challenge is automatically picked
    from the miner-test-data/ directory.
    """
    from babelbit.cli.local_runner import main as run_local
    rc = run_local(
        challenge=challenge,
        miner_urls=list(miner_url) if miner_url else None,
        output_dir=output_dir,
        max_challenges=max_challenges,
        max_dialogues=max_dialogues,
        timeout=timeout,
    )
    sys.exit(rc)


@cli.command("test-miner")
@click.option(
    "--challenge",
    required=True,
    help="Path to a challenge JSON file or directory of challenge files.",
)
@click.option(
    "--miner-url",
    default="http://localhost:8091",
    show_default=True,
    help="Base URL of the local miner server.",
)
@click.option(
    "--output-dir",
    default="local_test_output",
    show_default=True,
    help="Directory for JSONL logs and score output.",
)
@click.option(
    "--max-challenges",
    type=int,
    default=None,
    help="Max challenge files to process (random selection when given a directory).",
)
@click.option(
    "--max-dialogues",
    type=int,
    default=None,
    help="Max dialogues per challenge (for quick tests).",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Prediction request timeout in seconds.",
)
@click.option(
    "--skip-scoring",
    is_flag=True,
    help="Skip scoring, only produce JSONL logs.",
)
def test_miner_cmd(challenge, miner_url, output_dir, max_challenges, max_dialogues, timeout, skip_scoring):
    """Run local miner scoring test against challenge data."""
    from babelbit.miner.tests.local_test_challenge import main as run_test
    rc = run_test(
        challenge=challenge,
        miner_url=miner_url,
        output_dir=output_dir,
        max_challenges=max_challenges,
        max_dialogues=max_dialogues,
        timeout=timeout,
        skip_scoring=skip_scoring,
    )
    sys.exit(rc)


# TODO: remove this later
@cli.command("test-metagraph")
def test_metagraph_cmd():
    run(test_metagraph())


if __name__ == "__main__":
    basicConfig(
        level=INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cli()
