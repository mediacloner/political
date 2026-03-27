"""
TabbyAPI HTTP client — wraps the OpenAI-compatible endpoints exposed by TabbyAPI.
Handles model load/unload and chat completion.
"""

import requests
import time
from typing import Optional


class TabbyClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._current_model: Optional[str] = None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self, model_name: str, model_path: str, max_seq_len: int = 8192) -> None:
        """Load a model into VRAM. Blocks until the model is ready."""
        if self._current_model == model_name:
            return  # Already loaded

        payload = {
            "name": model_name,
            "max_seq_len": max_seq_len,
        }
        resp = requests.post(
            f"{self.base_url}/v1/model/load",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._current_model = model_name

    def unload_model(self) -> None:
        """Evict the current model from VRAM."""
        if self._current_model is None:
            return
        resp = requests.post(
            f"{self.base_url}/v1/model/unload",
            headers=self.headers,
            timeout=30,
        )
        resp.raise_for_status()
        self._current_model = None

    def swap_model(
        self,
        model_name: str,
        model_path: str,
        max_seq_len: int = 8192,
        verbose: bool = True,
    ) -> float:
        """Unload current model and load the next one. Returns swap time in seconds."""
        if self._current_model == model_name:
            return 0.0

        t0 = time.time()
        if verbose:
            print(f"  [swap] {self._current_model or 'none'} → {model_name}", flush=True)

        self.unload_model()
        self.load_model(model_name, model_path, max_seq_len)

        elapsed = time.time() - t0
        if verbose:
            print(f"  [swap] done in {elapsed:.1f}s", flush=True)
        return elapsed

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 1024,
    ) -> str:
        """Send a chat completion request. Returns the assistant message content."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def is_alive(self) -> bool:
        """Check if TabbyAPI is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model
