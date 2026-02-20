# Babelbit Local Testing Guide

Run local miners and a validator to evaluate prediction quality without a Bittensor wallet, Chutes account, or any external services. Supports single-miner and multi-miner setups.

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

## 4. Start local miners

### Single miner

Open a terminal and run:

```bash
./run_local_miner.sh
```

Wait until you see:

```
INFO:     Uvicorn running on 0.0.0.0:8091
INFO:     Model loaded successfully
```

### Multiple miners

To compare different models side by side, start each miner in its own terminal on a different port:

**Terminal 1 -- Miner A (distilgpt2 on port 8091):**

```bash
MINER_AXON_PORT=8091 MINER_MODEL_ID=distilgpt2 ./run_local_miner.sh
```

**Terminal 2 -- Miner B (gpt2 on port 8092):**

```bash
MINER_AXON_PORT=8092 MINER_MODEL_ID=gpt2 ./run_local_miner.sh
```

The script auto-kills any leftover process on the target port before starting.

### Miner environment variables

| Variable | Default | Description |
|---|---|---|
| `MINER_DEV_MODE` | `1` | Skip Bittensor signature verification |
| `MINER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MINER_MODEL_ID` | `distilgpt2` | HuggingFace model ID |
| `MINER_AXON_PORT` | `8091` | HTTP port |

## 5. Run the local validator

The validator sends the **same challenge to every miner concurrently** (just like mainnet), then scores each miner independently.

### Single miner (default)

```bash
./run_local_validator.sh
```

This connects to `http://localhost:8091` by default.

### Multiple miners

Pass each miner URL with `--miner-url`:

```bash
./run_local_validator.sh \
    --miner-url http://localhost:8091 \
    --miner-url http://localhost:8092
```

Or use the `MINER_URLS` environment variable (comma-separated):

```bash
MINER_URLS=http://localhost:8091,http://localhost:8092 ./run_local_validator.sh
```

Both methods are equivalent. Do not mix them -- if `--miner-url` flags are passed, `MINER_URLS` is ignored.

### Validator options

```
--challenge PATH       Specific challenge file or directory (default: random from miner-test-data/)
--miner-url URL        Miner endpoint (repeatable for multiple miners)
--output-dir DIR       Output directory (default: local_test_output)
--max-challenges N     Number of random challenges to run (default: 1)
--max-dialogues N      Max dialogues per challenge (default: all)
--timeout SECONDS      Prediction request timeout (default: 30.0)
```

### Examples

```bash
# 1 random challenge, single miner (simplest)
./run_local_validator.sh

# 2 miners, 1 random challenge
./run_local_validator.sh \
    --miner-url http://localhost:8091 \
    --miner-url http://localhost:8092

# 2 miners, 5 random challenges
MINER_URLS=http://localhost:8091,http://localhost:8092 \
    ./run_local_validator.sh --max-challenges 5

# specific challenge file, 2 miners
./run_local_validator.sh \
    --miner-url http://localhost:8091 \
    --miner-url http://localhost:8092 \
    --challenge ../miner-test-data/en/npr/03/en-npr-003026.json

# limit dialogues per challenge
./run_local_validator.sh --max-challenges 3 --max-dialogues 2
```

## 6. Understanding the output

### Multi-miner output

After a run completes you will see per-miner scores:

```
================================================================
Babelbit Local Validator (mainnet-equivalent flow)
================================================================
  Miners (2):
    [0] http://localhost:8091
    [1] http://localhost:8092
  Challenge:    /path/to/miner-test-data
  Output dir:   local_test_output

Found 1 challenge file(s)

Challenge en-npr-003026…  (5 dialogue(s), 2 miner(s))
----------------------------------------------------------------
  Scored 5 dialogue(s) for 2 miner(s)
    Miner [0] http://localhost:8091: U = 0.1234
    Miner [1] http://localhost:8092: U = 0.0987

================================================================
RESULTS SUMMARY
================================================================
  Miner [0] http://localhost:8091
    Mean U across 1 challenge(s): 0.1234
  Miner [1] http://localhost:8092
    Mean U across 1 challenge(s): 0.0987
================================================================
```

Unhealthy miners (not reachable) are automatically skipped with a warning.

### Output files

```
local_test_output/
├── logs/       # JSONL files with step-by-step predictions (per miner)
└── scores/     # JSON files with per-dialogue scoring breakdown (per miner)
```

### Scoring metrics

The scorer uses the same production pipeline as mainnet validators:

- **Semantic similarity** -- cosine similarity of sentence embeddings (`mxbai-embed-large-v1`)
- **Lexical similarity** -- Levenshtein distance normalization
- **U score** -- combined metric per utterance; higher is better

## 7. Using a custom model

The miner uses HuggingFace `AutoModelForCausalLM`, so any causal language model (GPT-style) works. There are three ways to load your own model.

### From HuggingFace Hub

Point `MINER_MODEL_ID` at your Hub repo:

```bash
# Public model
MINER_MODEL_ID=your-username/your-model ./run_local_miner.sh

# Private model (requires auth token)
HF_TOKEN=hf_xxxxx MINER_MODEL_ID=your-username/your-private-model ./run_local_miner.sh
```

### From a local directory

If you have model weights saved locally (a directory containing `config.json`, `model.safetensors` or `pytorch_model.bin`, `tokenizer.json`, etc.), pass the **absolute path**:

```bash
MINER_MODEL_ID=/path/to/your/local-model ./run_local_miner.sh
```

To save a fine-tuned model for this purpose:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("base-model")
tokenizer = AutoTokenizer.from_pretrained("base-model")

# ... fine-tune or modify ...

model.save_pretrained("/path/to/your/local-model")
tokenizer.save_pretrained("/path/to/your/local-model")
```

### Specific revision or branch

```bash
MINER_MODEL_ID=your-username/your-model MINER_MODEL_REVISION=v2 ./run_local_miner.sh
```

### Model environment variables

| Variable | Example | Description |
|---|---|---|
| `MINER_MODEL_ID` | `your-username/your-model` or `/absolute/path` | HuggingFace ID or local path |
| `MINER_MODEL_REVISION` | `main`, `v2` | Branch/tag on HuggingFace |
| `HF_TOKEN` | `hf_xxxxx` | Auth token for private models |
| `MINER_DEVICE` | `cpu` or `cuda` | Device to load the model on |

Compatible model families include GPT-2, LLaMA, Mistral, Phi, Qwen, and any other `AutoModelForCausalLM`-compatible architecture.

### Using your own backend (proxy mode)

If you have your own inference server, you can run the miner in **backend proxy mode**. The miner will forward every `/predict` request to your backend instead of loading a model locally. No GPU or model download required.

Set `MINER_BACKEND_URL` to activate:

```bash
MINER_BACKEND_URL=http://localhost:5000 ./run_local_miner.sh
```

Your backend must implement a `POST /predict` endpoint that accepts this JSON body:

```json
{
  "index": "session-uuid",
  "step": 3,
  "prefix": "The quick brown",
  "context": "Previous utterance text",
  "done": false,
  "prediction": ""
}
```

And return a JSON response with the predicted full utterance:

```json
{
  "prediction": "The quick brown fox jumps over the lazy dog"
}
```

Optionally, your backend can also expose `GET /healthz` (returning HTTP 200) for health checks.

#### Minimal Flask backend example

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prefix = data.get("prefix", "")
    context = data.get("context", "")

    # Your custom prediction logic here
    prediction = my_model_predict(prefix, context)

    return jsonify({"prediction": prediction})

@app.route("/healthz")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

#### Backend environment variables

| Variable | Example | Description |
|---|---|---|
| `MINER_BACKEND_URL` | `http://localhost:5000` | Your backend URL; activates proxy mode |
| `MINER_BACKEND_TIMEOUT` | `30` | Request timeout in seconds (default: 30) |

#### Full workflow with a custom backend

```bash
# Terminal 1: start your backend
python my_backend.py  # serving on port 5000

# Terminal 2: start the miner in proxy mode
MINER_BACKEND_URL=http://localhost:5000 ./run_local_miner.sh

# Terminal 3: run the validator
./run_local_validator.sh
```

The validator sees the miner on port 8091 as usual -- it has no knowledge that the miner is proxying to your backend.

## 8. Quick-reference cheat sheet

### Single miner

```bash
# Terminal 1: start miner
./run_local_miner.sh

# Terminal 2: run validator
./run_local_validator.sh
```

### Multi-miner comparison

```bash
# Terminal 1: miner A
MINER_AXON_PORT=8091 MINER_MODEL_ID=distilgpt2 ./run_local_miner.sh

# Terminal 2: miner B
MINER_AXON_PORT=8092 MINER_MODEL_ID=gpt2 ./run_local_miner.sh

# Terminal 3: validator against both
./run_local_validator.sh \
    --miner-url http://localhost:8091 \
    --miner-url http://localhost:8092
```

### Useful commands

```bash
# Check a miner's health
curl http://localhost:8091/healthz

# Run with a different model
MINER_MODEL_ID=gpt2 ./run_local_miner.sh

# Run 10 random challenges against 2 miners
MINER_URLS=http://localhost:8091,http://localhost:8092 \
    ./run_local_validator.sh --max-challenges 10
```

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot reach miner at ...` | Make sure the miner is running and the port matches |
| `No healthy miners found` | None of the miner URLs responded; start miners first |
| `address already in use` | `run_local_miner.sh` auto-kills stale processes; or manually run `fuser -k PORT/tcp` |
| `No challenge files found` | Check that `miner-test-data/` exists as a sibling directory, or pass `--challenge` explicitly |
| `CUDA out of memory` | Switch to a smaller model (`distilgpt2`) or use `MINER_DEVICE=cpu` |
| 3 miners shown when expecting 2 | Pass `--miner-url` flags directly instead of mixing with `MINER_URLS` env var |
| Slow first run | The model and scoring embedder are downloaded on first use; subsequent runs are cached |
