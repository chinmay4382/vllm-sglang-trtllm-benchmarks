"""
Concurrent load tester for an OpenAI-compatible vLLM inference endpoint.

Streams responses token-by-token to capture TTFT and ITL accurately.
"""

import asyncio
import logging
import time
from typing import List, Optional

import httpx

from benchmark.metrics import BenchmarkMetrics, RequestResult, compute_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt corpus — overridable via config
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning in machine learning.",
    "Write a Python function that implements binary search on a sorted list.",
    "What are the main causes of climate change and what can individuals do to help?",
    "Describe the architecture of a transformer neural network.",
    "How does the TCP/IP protocol stack work? Explain each layer.",
    "Write a SQL query to find the top 5 customers by total purchase amount.",
    "What is the difference between a mutex and a semaphore?",
    "Explain the concept of gradient descent in optimization.",
    "Describe the CAP theorem and its implications for distributed systems.",
    "Write a regular expression to validate an email address.",
    "How does garbage collection work in modern programming languages?",
    "Explain the difference between REST and GraphQL APIs.",
    "What is the time complexity of quicksort in the average and worst cases?",
    "Describe the SOLID principles of object-oriented design.",
    "How does HTTPS ensure secure communication over the internet?",
    "Write a function to detect if a linked list has a cycle.",
    "Explain the concept of eventual consistency in distributed databases.",
    "What are the differences between Docker containers and virtual machines?",
    "Describe how a hash table handles collisions.",
    "What is the difference between process and thread in operating systems?",
]


class LoadTester:
    """
    Sends concurrent streaming chat-completion requests to a vLLM endpoint
    and collects per-request timing data.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 120.0,
        prompts: Optional[List[str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.prompts = prompts or DEFAULT_PROMPTS

    async def _send_request(
        self,
        client: httpx.AsyncClient,
        request_id: int,
        prompt: str,
    ) -> RequestResult:
        """Issue a single streaming request and record timing checkpoints."""
        result = RequestResult(
            request_id=request_id,
            prompt=prompt,
            start_time=time.perf_counter(),
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                token_count = 0

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break

                    import json
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        if result.first_token_time is None:
                            result.first_token_time = time.perf_counter()
                        token_count += len(content.split())  # approximate

                    # Capture usage if available in the final chunk
                    usage = chunk.get("usage")
                    if usage:
                        result.total_tokens = usage.get("completion_tokens", token_count)
                        result.prompt_tokens = usage.get("prompt_tokens", 0)

                if result.total_tokens == 0:
                    result.total_tokens = token_count

        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            logger.warning("Request %d failed: %s", request_id, result.error)
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            logger.warning("Request %d error: %s", request_id, result.error)
        finally:
            result.end_time = time.perf_counter()

        return result

    async def _run_concurrency_level(
        self,
        concurrency: int,
        num_requests: int,
    ) -> tuple[List[RequestResult], float]:
        """
        Run `num_requests` prompts with up to `concurrency` in-flight at once.

        Returns (results, wall_time).
        """
        semaphore = asyncio.Semaphore(concurrency)
        prompts = (self.prompts * ((num_requests // len(self.prompts)) + 1))[:num_requests]

        async def bounded(request_id: int, prompt: str) -> RequestResult:
            async with semaphore:
                return await self._send_request(client, request_id, prompt)

        async with httpx.AsyncClient() as client:
            wall_start = time.perf_counter()
            tasks = [bounded(i, p) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)
            wall_time = time.perf_counter() - wall_start

        return list(results), wall_time

    def run(
        self,
        concurrency_levels: List[int],
        num_requests: int = 50,
        dataset_name: str = "default",
    ) -> List[BenchmarkMetrics]:
        """
        Execute a sweep across all concurrency levels and return aggregated metrics.

        Args:
            concurrency_levels: List of concurrent-user counts to test (e.g. [1, 5, 10, 20]).
            num_requests: Total requests per concurrency level.
            dataset_name: Label for the prompt dataset.

        Returns:
            List of BenchmarkMetrics, one per concurrency level.
        """
        all_metrics: List[BenchmarkMetrics] = []

        for concurrency in concurrency_levels:
            logger.info(
                "Starting load test | model=%s concurrency=%d requests=%d",
                self.model,
                concurrency,
                num_requests,
            )
            results, wall_time = asyncio.run(
                self._run_concurrency_level(concurrency, num_requests)
            )
            metrics = compute_metrics(
                results=results,
                model_name=self.model,
                concurrency=concurrency,
                dataset_name=dataset_name,
                wall_time=wall_time,
            )
            all_metrics.append(metrics)

        return all_metrics


def run_load_test(
    base_url: str,
    model: str,
    concurrency_levels: List[int],
    num_requests: int = 50,
    max_tokens: int = 256,
    dataset_name: str = "default",
    prompts: Optional[List[str]] = None,
) -> List[BenchmarkMetrics]:
    """
    Convenience entry point for running a full load test sweep.

    Returns a list of BenchmarkMetrics (one per concurrency level).
    """
    tester = LoadTester(
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        prompts=prompts,
    )
    return tester.run(
        concurrency_levels=concurrency_levels,
        num_requests=num_requests,
        dataset_name=dataset_name,
    )
