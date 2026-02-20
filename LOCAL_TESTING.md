# Babelbit Local Testing Guide

Run a local miner and validator to evaluate prediction quality without a Bittensor wallet, Chutes account, or any external services.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.13 | |
| [uv](https://docs.astral.sh/uv/) | latest | Python package manager |
| Git | any | to clone the repo |
| GPU (optional) | CUDA-capable | set `MINER_DEVICE=cuda`; defaults to `cpu` |

## 1. Clone and install dependencies

```bash
git clone <repo-url> babelbit_subnet_loifaith
cd babelbit_subnet_loifaith
uv sync
```

`uv sync` reads `pyproject.toml` and installs all runtime and dev dependencies into a local virtual environment managed by `uv`.

## 2. Prepare test data

Challenge files live in a sibling directory called `miner-test-data/`. The validator auto-discovers this directory when you omit `--challenge`.

```
your-workspace/
├── babelbit_subnet_loifaith/   # this repo
│   ├── run_local_miner.sh
│   ├── run_local_validator.sh
│   └── ...
└── miner-test-data/            # challenge JSON files
    └── en/npr/...
```

If you already have the test data elsewhere, pass the path explicitly with `--challenge`.

## 3. Configure the environment

Copy the example env file:

```bash
cp env.test.example .env
```

Default `.env` values (edit as needed):

```bash
MINER_DEV_MODE=1          # bypass Bittensor header verification
MINER_DEVICE=cpu          # 'cpu' or 'cuda'
MINER_MODEL_ID=distilgpt2 # any HuggingFace causal-LM model ID
MINER_AXON_PORT=8091      # miner HTTP port
```

## 4. Start the local miner

Open a terminal and run:

```bash
./run_local_miner.sh
```

Or equivalently:

```bash
MINER_DEV_MODE=1 MINER_DEVICE=cpu MINER_MODEL_ID=distilgpt2 \
    uv run python babelbit/miner/serve_miner.py
```

Wait until you see output like:

```
INFO:     Uvicorn running on 0.0.0.0:8091
INFO:     Model loaded successfully
```

### Miner environment variables

| Variable | Default | Description |
|---|---|---|
| `MINER_DEV_MODE` | `1` | Skip Bittensor signature verification |
| `MINER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MINER_MODEL_ID` | `distilgpt2` | HuggingFace model ID |
| `MINER_AXON_PORT` | `8091` | HTTP port |

Override inline:

```bash
MINER_MODEL_ID=gpt2 MINER_DEVICE=cuda ./run_local_miner.sh
```

## 5. Run the local validator

Open a **second terminal** and run:

```bash
./run_local_validator.sh
```

Or equivalently:

```bash
uv run python -c "from babelbit import cli; cli()" local-validate
```

By default this:
1. Auto-discovers `miner-test-data/`
2. Picks **1 random** challenge file
3. Simulates the utterance engine (token-by-token revelation)
4. Sends each step to the local miner's `/predict` endpoint
5. Scores the predictions using the production scoring pipeline
6. Writes logs and scores to `local_test_output/`

### Validator options

```
--challenge PATH       Specific challenge file or directory (default: auto-detect)
--miner-url URL        Miner endpoint (default: http://localhost:8091)
--output-dir DIR       Output directory (default: local_test_output)
--max-challenges N     Number of random challenges to run (default: 1)
--max-dialogues N      Max dialogues per challenge (default: all)
--timeout SECONDS      Prediction request timeout (default: 30.0)
```

### Examples

```bash
# 1 random challenge (simplest)
./run_local_validator.sh

# 5 random challenges
./run_local_validator.sh --max-challenges 5

# specific challenge file
./run_local_validator.sh --challenge ../miner-test-data/en/npr/03/en-npr-003026.json

# limit dialogues per challenge
./run_local_validator.sh --max-challenges 3 --max-dialogues 2

# custom miner URL
MINER_URL=http://localhost:9000 ./run_local_validator.sh
```

## 6. Understanding the output

After a run completes you will see a summary like:

```
================================================================
Babelbit Local Validator (mainnet-equivalent flow)
================================================================
  Miner URL:    http://localhost:8091
  Challenge:    /path/to/miner-test-data
  Output dir:   local_test_output

Found 1 challenge file(s)

Challenge en-npr-003026…  (5 dialogue(s))
----------------------------------------------------------------
  Scored 5 dialogue(s) for 1 miner(s)
  Challenge mean U: 0.1234

================================================================
Overall mean U across 1 challenge(s): 0.1234
================================================================
```

### Output files

```
local_test_output/
├── logs/       # JSONL files with step-by-step predictions
└── scores/     # JSON files with per-dialogue scoring breakdown
```

### Scoring metrics

The scorer uses the same production pipeline as mainnet validators:

- **Semantic similarity** -- cosine similarity of sentence embeddings (`mxbai-embed-large-v1`)
- **Lexical similarity** -- Levenshtein distance normalization
- **U score** -- combined metric per utterance; higher is better

## 7. Quick-reference cheat sheet

```bash
# Terminal 1: start miner
./run_local_miner.sh

# Terminal 2: run validator (once miner is ready)
./run_local_validator.sh

# Check miner health manually
curl http://localhost:8091/healthz

# Run with a different model
MINER_MODEL_ID=gpt2 ./run_local_miner.sh
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot reach miner at ...` | Make sure the miner is running and the port matches |
| `No challenge files found` | Check that `miner-test-data/` exists as a sibling directory, or pass `--challenge` explicitly |
| `CUDA out of memory` | Switch to a smaller model (`distilgpt2`) or use `MINER_DEVICE=cpu` |
| `resume_download` TypeError | Already fixed; ensure you have the latest code |
| Slow first run | The model and scoring embedder are downloaded on first use; subsequent runs are cached |
