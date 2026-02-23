"""
Simple FastAPI-based miner server for Babelbit subnet.
Serves predictions via HTTP endpoint that validators can call directly.

Note: Run register_axon.py first to register your miner on-chain,
then run this script to serve predictions.
"""
import asyncio
import json
import logging
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

import torch
from fastapi import FastAPI, HTTPException, Request, Header, status
from pydantic import BaseModel
import uvicorn
from substrateinterface import Keypair

from openai import OpenAI

from babelbit.miner.model_loader import load_model_and_tokenizer
from babelbit.miner.utils import verify_bittensor_request
from babelbit.utils.bittensor_helpers import load_hotkey_keypair
from babelbit.utils.settings import get_settings

logger = logging.getLogger(__name__)

# Simple in-process cache for tokenized static prompt prefixes
_PROMPT_CACHE: dict[str, torch.Tensor] = {}
_MINER_HOTKEY_SS58: Optional[str] = None

# Optional path for per-step prediction logging (hotkey name is appended when writing)
_PREDICTION_LOG_PATH = os.getenv("MINER_PREDICTION_LOG", "miner_predictions.jsonl")
# Env file path from CLI (--envfile); used by get_settings() in this module
_ENV_FILE: Optional[str] = None


def _prediction_log_path_with_hotkey() -> str:
    """Return _PREDICTION_LOG_PATH with hotkey name appended before the extension (e.g. miner_predictions_default.jsonl)."""
    base = _PREDICTION_LOG_PATH
    hotkey_name = "unknown"
    try:
        hotkey_name = _settings().BITTENSOR_WALLET_HOT or "unknown"
    except Exception:
        pass
    # Safe for filenames: replace path separators and spaces
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(hotkey_name))
    if not safe:
        safe = "unknown"
    root, ext = os.path.splitext(base)
    return f"{root}_{safe}{ext}"


def _settings():
    """Settings loaded from _ENV_FILE when set (CLI --envfile)."""
    return get_settings(_ENV_FILE) if _ENV_FILE else get_settings()


def _get_backend_url() -> Optional[str]:
    """BACKEND_URL (or MINER_BACKEND_URL) from env file pointed by --envfile."""
    _settings()  # ensure env file is loaded
    url = (os.getenv("BACKEND_URL") or os.getenv("MINER_BACKEND_URL") or "").strip()
    return url or None


def predict_via_backend(
    backend_url: str,
    prompt: str,
    max_tokens: int = 24,
    model: Optional[str] = None,
) -> str:
    """
    Call an OpenAI-compatible completion API (e.g. SGLang) at backend_url.
    Returns the completion text only (no prefix).
    """
    base = backend_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    client = OpenAI(api_key="cpk_61697c3eeca84f3b8be5007365b561b3.4f3dc67df0ac5df49ec8ccb04b719490.k7ndwuvjSB5Mfu4cZR6tlbU44jn8J0tt", base_url=base)
    model_name = model or os.getenv("BACKEND_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    text = (response.choices[0].text or "").strip()
    return text


def _get_miner_hotkey_ss58() -> Optional[str]:
    """Load and cache this miner's hotkey SS58 address."""
    global _MINER_HOTKEY_SS58
    if _MINER_HOTKEY_SS58:
        return _MINER_HOTKEY_SS58
    try:
        settings = _settings()
        keypair = load_hotkey_keypair(settings.BITTENSOR_WALLET_COLD, settings.BITTENSOR_WALLET_HOT)
        _MINER_HOTKEY_SS58 = keypair.ss58_address
        return _MINER_HOTKEY_SS58
    except Exception as e:
        logger.warning(f"Unable to load miner hotkey for request verification: {e}")
        return None


class BBUtteranceEvaluation(BaseModel):
    """Evaluation result for utterance prediction."""
    lexical_similarity: float = 0.0
    semantic_similarity: float = 0.0
    earliness: float = 0.0
    u_step: float = 0.0


class PredictRequest(BaseModel):
    """Request schema matching chute template."""
    index: str  # UUID session identifier
    step: int
    prefix: str
    context: str = ""
    done: bool = False
    ground_truth: str | None = None
    prediction: str = ""
    evaluation: BBUtteranceEvaluation | None = None


class PredictResponse(BaseModel):
    """Simple response schema expected by validator."""
    prediction: str


class BabelbitMiner:
    """Miner that serves predictions using a Hugging Face model."""
    
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the miner with a model.
        
        Args:
            model_id: Hugging Face model ID (e.g., "meta-llama/Llama-2-7b-hf")
            revision: Model revision/branch to use
            cache_dir: Directory for model cache
            device: Device to load model on ("cuda" or "cpu")
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Determine device and dtype
        self.device = self._pick_device() if device == "cuda" else torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # Model and tokenizer loaded on demand
        self._model = None
        self._tokenizer = None
        self._model_lock = asyncio.Lock()
        self._model_moved = False
        
        logger.info(f"Initialized BabelbitMiner with model: {model_id}")
        logger.info(f"Target device: {self.device}, dtype: {self.dtype}")
    
    def _pick_device(self) -> torch.device:
        """Select best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def _get_env_int(self, name: str, default: int) -> int:
        """Get integer from environment variable."""
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default
    
    def _get_env_float(self, name: str, default: float) -> float:
        """Get float from environment variable."""
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default

    def _use_chat_template(self) -> bool:
        """Whether to use the tokenizer's chat template. Base models (e.g. Llama-3.1-8B) should not."""
        raw = os.getenv("CHUTE_USE_CHAT_TEMPLATE", "").strip().lower()
        if raw in ("0", "false", "no"):
            return False
        if raw in ("1", "true", "yes"):
            return True
        # Auto-detect: only use chat template for Instruct/Chat-tuned models
        model_lower = (self.model_id or "").lower()
        return "instruct" in model_lower or "chat" in model_lower
    
    def _prepare_inputs(self, prompt: str) -> torch.Tensor:
        """Tokenize prompt with caching of static system+instruction part."""
        try:
            # Simple cache key for tiny prompts
            if len(prompt) < 256:
                inputs = self._tokenizer.encode(prompt, return_tensors="pt")
                if inputs.numel() == 0:
                    raise ValueError("Empty tokenization result")
                return inputs.to(self.device)
            
            # Heuristic: find last occurrence of static instruction
            marker = "Continue the utterance"
            idx = prompt.rfind(marker)
            if idx != -1:
                static_part = prompt[:idx]
                cache_key = static_part
                dynamic_part = prompt[idx:]
                
                if cache_key in _PROMPT_CACHE:
                    static_ids = _PROMPT_CACHE[cache_key]
                else:
                    static_ids = self._tokenizer.encode(static_part, return_tensors="pt")
                    if static_ids.numel() == 0:
                        raise ValueError("Empty static tokenization result")
                    _PROMPT_CACHE[cache_key] = static_ids
                
                dyn_ids = self._tokenizer.encode(dynamic_part, return_tensors="pt")
                if dyn_ids.numel() == 0:
                    raise ValueError("Empty dynamic tokenization result")
                
                # Concatenate tokens
                if dyn_ids.size(1) > 1:
                    full = torch.cat([static_ids, dyn_ids[:, 1:]], dim=1)
                else:
                    full = static_ids
                
                if full.numel() == 0:
                    raise ValueError("Empty concatenated tensor")
                return full.to(self.device)
            
            # Fallback: no split
            inputs = self._tokenizer.encode(prompt, return_tensors="pt")
            if inputs.numel() == 0:
                raise ValueError("Empty fallback tokenization result")
            return inputs.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in _prepare_inputs: {str(e)}")
            # Emergency fallback - create a simple tensor with EOS token
            if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
                fallback_tensor = torch.tensor([[self._tokenizer.eos_token_id]], dtype=torch.long)
            else:
                fallback_tensor = torch.tensor([[1]], dtype=torch.long)
            return fallback_tensor.to(self.device)
    
    async def load(self):
        """Load model and tokenizer (called once at startup)."""
        async with self._model_lock:
            if self._model is None:
                logger.info(f"Loading model {self.model_id}...")
                self._model, self._tokenizer = await asyncio.to_thread(
                    load_model_and_tokenizer,
                    model_id=self.model_id,
                    revision=self.revision,
                    cache_dir=self.cache_dir,
                    device=self.device,
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                )
                logger.info(f"Model loaded successfully on {self.device}")
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Generate prediction for the given prefix and context.
        
        Args:
            request: Prediction request with prefix and context
            
        Returns:
            PredictResponse with generated text (just the prediction string)
        """
        try:
            # Ensure model is loaded
            if self._model is None:
                await self.load()
            
            if not request.prefix:
                logger.warning("No prefix provided, returning empty prediction")
                return PredictResponse(prediction="")

            backend_url = _get_backend_url()
            if backend_url:
                # Use OpenAI-compatible backend (e.g. SGLang) instead of local model
                context = request.context
                if context:
                    context = re.sub(r"\s*\(as\s+[^)]+\)", "", context).strip()
                if context:
                    prompt = f"Context:\n{context}\n\nContinue the utterance that begins with:\n{request.prefix}"
                else:
                    prompt = f"Continue the utterance that begins with:\n{request.prefix}"
                max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
                try:
                    completion = await asyncio.to_thread(
                        predict_via_backend,
                        backend_url,
                        prompt,
                        max_tokens=max_new_tokens,
                    )
                except Exception as e:
                    logger.error(f"Backend request failed: {e}")
                    completion = os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
                full_prediction = request.prefix + " " + (completion or "").strip()
                if not (completion or "").strip():
                    full_prediction = request.prefix + " " + os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
                logger.info(f"Backend generated: {full_prediction[:100]}...")
                try:
                    log_record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "index": request.index,
                        "step": request.step,
                        "done": request.done,
                        "prefix": request.prefix,
                        "context": request.context,
                        "ground_truth": request.ground_truth,
                        "raw_prediction": completion or "",
                        "full_prediction": full_prediction,
                    }
                    if request.evaluation is not None:
                        log_record["evaluation"] = request.evaluation.model_dump()
                    with open(_prediction_log_path_with_hotkey(), "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                except Exception as log_err:
                    logger.error(f"Failed to write prediction log: {log_err}")
                return PredictResponse(prediction=full_prediction)

            logger.info(f"Generating prediction for prefix: '{request.prefix[:60]}{'...' if len(request.prefix) > 60 else ''}'")
            if request.context:
                _ctx_preview = request.context[:80] + "..." if len(request.context) > 80 else request.context
                logger.info("Using context: %s chars (%s)", len(request.context), _ctx_preview.replace("%", "%%"))
            
            # Sanitize script-style metadata from context (e.g. "(as Character)", "(as Self)")
            context = request.context
            if context:
                context = re.sub(r"\s*\(as\s+[^)]+\)", "", context)
                context = context.strip()
            
            # Use chat template only for Instruct/Chat models; base models (e.g. Llama-3.1-8B)
            # are not trained on chat format, so we use a plain continuation prompt instead.
            use_chat_template = self._use_chat_template()
            try:
                if use_chat_template and hasattr(self._tokenizer, 'apply_chat_template'):
                    # Instruct/Chat models: use instruction + chat template
                    system_msg = (
                        "You are a helpful assistant that completes the current utterance naturally and succinctly. "
                        "Return only the completed utterance text without quotes or extra commentary."
                    )
                    if context:
                        user_msg = f"Context:\n{context}\n\nContinue the utterance that begins with:\n{request.prefix}"
                    else:
                        user_msg = f"Continue the utterance that begins with:\n{request.prefix}"
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Base completion models: prompt = text to continue (predict words after prefix)
                    if context:
                        prompt = f"Context:\n{context}\n\n{request.prefix}"
                    else:
                        prompt = request.prefix
            except Exception:
                # Simple fallback if chat template fails
                prompt = f"{request.prefix}"
            
            # Move model to device lazily (only first call)
            if not self._model_moved:
                try:
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info(f"Model moved to {self.device}")
                    self._model_moved = True
                except Exception as e:
                    logger.error(f"Error moving model to device: {str(e)}")
                    # Fallback to CPU
                    self.device = torch.device("cpu")
                    self._model.to(self.device)
                    self._model.eval()
                    logger.info("Fell back to CPU device")
                    self._model_moved = True
            
            # Tokenize with caching
            try:
                inputs = await asyncio.to_thread(self._prepare_inputs, prompt)
                
                # Validate input tensor
                if inputs.dim() != 2 or inputs.size(0) != 1:
                    raise ValueError(f"Invalid input tensor shape: {inputs.shape}")
                # Use len(tokenizer) not vocab_size: Llama etc. have added/special tokens with IDs >= vocab_size
                max_token_id = len(self._tokenizer) - 1
                if torch.any(inputs > max_token_id) or torch.any(inputs < 0):
                    raise ValueError("Input contains invalid token IDs")
                    
            except Exception as e:
                logger.error(f"Error preparing inputs: {str(e)}")
                # Create safe fallback input
                fallback_text = request.prefix[:50] if request.prefix else "Hello"
                inputs = self._tokenizer.encode(
                    fallback_text, return_tensors="pt", max_length=512, truncation=True
                )
                inputs = inputs.to(self.device)
            
            # Get generation parameters from environment
            max_new_tokens = self._get_env_int("CHUTE_MAX_NEW_TOKENS", 24)
            temperature = self._get_env_float("CHUTE_TEMPERATURE", 0.7)
            top_p = self._get_env_float("CHUTE_TOP_P", 0.95)
            top_k = self._get_env_int("CHUTE_TOP_K", 50)
            do_sample = os.getenv("CHUTE_DO_SAMPLE", "1") not in ("0", "false", "False")
            
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            # Check for early exit if prefix already ends with EOS
            if self._tokenizer.eos_token and request.prefix.strip().endswith(self._tokenizer.eos_token):
                return PredictResponse(prediction="")
            
            # Generate prediction
            try:
                def generate():
                    with torch.no_grad():
                        if self.device.type == "cuda":
                            with torch.autocast(device_type="cuda", enabled=True):
                                return self._model.generate(inputs, **gen_kwargs)
                        else:
                            return self._model.generate(inputs, **gen_kwargs)
                
                outputs = await asyncio.to_thread(generate)
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error during generation: {str(e)}")
                    # Try on CPU as fallback
                    inputs_cpu = inputs.cpu()
                    self._model.cpu()
                    self.device = torch.device("cpu")
                    
                    with torch.no_grad():
                        outputs = self._model.generate(inputs_cpu, **gen_kwargs)
                else:
                    raise e
            
            # Decode the generated text
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part (remove the input prompt)
            if generated_text.startswith(prompt):
                prediction = generated_text[len(prompt):].strip()
            else:
                prediction = generated_text.strip()
            
            # Clean up the prediction
            prediction = prediction.replace("System:", "").replace("User:", "").replace("Assistant:", "").strip()
            
            # If prediction contains the original prefix, extract just the completion
            if request.prefix in prediction and prediction != request.prefix:
                prefix_pos = prediction.find(request.prefix)
                if prefix_pos != -1:
                    after_prefix = prediction[prefix_pos + len(request.prefix):].strip()
                    if after_prefix:
                        prediction = after_prefix
            
            # Ensure we have some prediction
            if not prediction or prediction.strip() == "" or prediction.strip() == request.prefix.strip():
                prediction = os.getenv("CHUTE_FALLBACK_COMPLETION", "...")
            
            # Return full utterance (prefix + prediction)
            full_prediction = request.prefix + ' ' + prediction
            
            logger.info(f"Generated: {full_prediction[:100]}...")

            # Log per-step prediction to external file (JSONL)
            try:
                log_record = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "index": request.index,
                    "step": request.step,
                    "done": request.done,
                    "prefix": request.prefix,
                    "context": request.context,
                    "ground_truth": request.ground_truth,
                    "raw_prediction": prediction,
                    "full_prediction": full_prediction,
                }
                # Include evaluation details if available
                if request.evaluation is not None:
                    log_record["evaluation"] = request.evaluation.model_dump()

                log_path = _prediction_log_path_with_hotkey()
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            except Exception as log_err:
                logger.error(f"Failed to write prediction log: {log_err}")
            
            return PredictResponse(prediction=full_prediction)
            
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            logger.error(format_exc())
            # Return empty prediction on error
            return PredictResponse(prediction="")


# Global miner instance
miner_instance: Optional[BabelbitMiner] = None


async def startup():
    """FastAPI startup event handler."""
    global miner_instance
    
    settings = _settings()
    
    # Get model configuration
    model_id = settings.MINER_MODEL_ID
    revision = getattr(settings, 'MINER_MODEL_REVISION', None)
    cache_dir = settings.BABELBIT_CACHE_DIR / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Quantization settings
    load_in_8bit = getattr(settings, 'MINER_LOAD_IN_8BIT', False)
    load_in_4bit = getattr(settings, 'MINER_LOAD_IN_4BIT', False)
    device = getattr(settings, 'MINER_DEVICE', 'cuda')
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Revision: {revision or 'main'}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
    logger.info(f"Device: {device}")
    logger.info("")

    backend_url = _get_backend_url()
    if backend_url:
        logger.info(f"Using backend URL: {backend_url} (skipping local model load)")
        logger.info("")

    # Create and load miner
    miner_instance = BabelbitMiner(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    if not backend_url:
        logger.info("Loading model...")
        try:
            await miner_instance.load()
            logger.info("✅ Model loaded successfully")
            logger.info("")
        except Exception as e:
            logger.error("❌ Failed to load model!")
            logger.error(f"   Error: {e}")
            logger.error("")
            logger.error("Common fixes:")
            logger.error("  1. For gated models (Llama, etc): Set HF_TOKEN environment variable")
            logger.error("     export HF_TOKEN=your_huggingface_token")
            logger.error("  2. Check model ID is correct and you have access")
            logger.error("  3. Ensure you have enough disk space and RAM/VRAM")
            raise
    else:
        logger.info("✅ Backend mode: predictions will use remote API")
        logger.info("")


# Create FastAPI app
app = FastAPI(title="Babelbit Miner", on_startup=[startup])


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.get("/health")
async def health_alt():
    """Alternative health check endpoint."""
    return {
        "status": "healthy",
        "model": miner_instance.model_id if miner_instance else "not_loaded",
        "model_loaded": miner_instance is not None and miner_instance._model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    request: PredictRequest,
    http_request: Request,
):
    """
    Prediction endpoint with Bittensor protocol verification.
    
    Validates incoming requests from validators using cryptographic signatures
    and nonce-based replay attack prevention.
    
    Set MINER_DEV_MODE=1 to bypass verification for local testing.
    """
    if miner_instance is None:
        raise HTTPException(status_code=503, detail="Miner not initialized")
    
    # Check if dev mode is enabled (bypass verification)
    settings = _settings()
    dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
    
    if dev_mode:
        logger.info("🔓 Dev mode enabled - bypassing Bittensor verification")
        return await miner_instance.predict(request)
    
    # Extract Bittensor headers
    headers = http_request.headers
    
    # Get required Bittensor protocol headers
    dendrite_hotkey = headers.get("bt_header_dendrite_hotkey")
    dendrite_nonce = headers.get("bt_header_dendrite_nonce")
    dendrite_signature = headers.get("bt_header_dendrite_signature")
    dendrite_uuid = headers.get("bt_header_dendrite_uuid")
    axon_hotkey = headers.get("bt_header_axon_hotkey")
    body_hash = headers.get("computed_body_hash", "")
    timeout_str = headers.get("timeout", "12.0")
    miner_hotkey = _get_miner_hotkey_ss58()
    
    # Check if this is a Bittensor protocol request
    is_bittensor_request = all([
        dendrite_hotkey,
        dendrite_nonce,
        dendrite_signature,
        dendrite_uuid,
        axon_hotkey,
    ])
    
    if is_bittensor_request:
        # Ensure the request is intended for this miner's hotkey
        if miner_hotkey and axon_hotkey and axon_hotkey != miner_hotkey:
            logger.warning(
                "Rejecting request: target hotkey mismatch (expected %s, got %s)",
                miner_hotkey,
                axon_hotkey,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Request not addressed to this miner hotkey",
            )

        # Verify the request using Bittensor protocol
        try:
            timeout = float(timeout_str)
        except ValueError:
            timeout = 12.0
        
        is_valid, error_msg = verify_bittensor_request(
            dendrite_hotkey=dendrite_hotkey,
            dendrite_nonce=dendrite_nonce,
            dendrite_signature=dendrite_signature,
            dendrite_uuid=dendrite_uuid,
            axon_hotkey=axon_hotkey,
            body_hash=body_hash,
            timeout=timeout,
        )
        
        if not is_valid:
            logger.warning(f"Request verification failed from {dendrite_hotkey[:8]}...: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Request verification failed: {error_msg}"
            )
        
        logger.info(f"✅ Verified request from validator: {dendrite_hotkey[:8]}...")
    else:
        # Non-Bittensor request - check dev mode
        settings = _settings()
        dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
        
        if not dev_mode:
            # In production mode, reject requests without Bittensor headers
            logger.warning("Rejecting request without Bittensor headers (production mode)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bittensor protocol headers required"
            )
        
        # Allow non-Bittensor requests only in dev mode
        logger.info("Processing request without Bittensor verification (dev mode)")
    
    # Process the prediction request
    return await miner_instance.predict(request)


async def main():
    """Main entry point for the miner server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = _settings()
    axon_port = settings.MINER_AXON_PORT
    
    logger.info("=" * 60)
    logger.info("Starting Babelbit Miner Server")
    logger.info("=" * 60)
    logger.info("")
    
    # Check dev mode
    dev_mode = getattr(settings, "MINER_DEV_MODE", False) or os.getenv("MINER_DEV_MODE", "0") in ("1", "true", "True")
    if dev_mode:
        logger.warning("🔓 DEV MODE ENABLED - Bittensor verification DISABLED")
        logger.warning("   This should ONLY be used for local testing!")
        logger.warning("   Set MINER_DEV_MODE=0 for production use.")
        logger.info("")
    else:
        logger.info("🔒 Bittensor verification enabled (production mode)")
        logger.info("")
    
    logger.info("⚠️  Make sure you've registered your axon first:")
    logger.info("   uv run python babelbit/miner/register_axon.py")
    logger.info("")
    
    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=axon_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    
    logger.info(f"🚀 Miner serving predictions on port {axon_port}")
    logger.info("   Press Ctrl+C to stop.")
    logger.info("")
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down miner server...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Miner from log (model for step 0,1; log lookup from step 2)",
    )
    parser.add_argument(
        "--envfile",
        type=str,
        metavar="PATH",
        help="Path to env file (default: .env)",
        default=None,
    )
    args = parser.parse_args()
    if args.envfile:
        globals()["_ENV_FILE"] = args.envfile
    asyncio.run(main())
