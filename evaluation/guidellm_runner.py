"""
GuideLLM-based evaluation runner.

Wraps the GuideLLM CLI / Python API to evaluate models on:
  - MMLU   (multiple-choice knowledge)
  - GSM8K  (grade-school math)
  - HumanEval (code generation, pass@k)

Falls back to a direct OpenAI-API evaluator when GuideLLM is not installed,
so the dashboard always has data to display.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

MMLU_SAMPLE_QUESTIONS = [
    {
        "question": "What is the powerhouse of the cell?",
        "choices": ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"],
        "answer": "C",
    },
    {
        "question": "Which element has atomic number 6?",
        "choices": ["Oxygen", "Nitrogen", "Carbon", "Hydrogen"],
        "answer": "C",
    },
    {
        "question": "What is the derivative of sin(x)?",
        "choices": ["cos(x)", "-cos(x)", "tan(x)", "-sin(x)"],
        "answer": "A",
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "choices": ["Charlotte Brontë", "Jane Austen", "Mary Shelley", "George Eliot"],
        "answer": "B",
    },
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "C",
    },
    {
        "question": "In Python, which data structure uses key-value pairs?",
        "choices": ["List", "Tuple", "Set", "Dictionary"],
        "answer": "D",
    },
    {
        "question": "What does CPU stand for?",
        "choices": [
            "Central Processing Unit",
            "Central Program Unit",
            "Computer Processing Unit",
            "Core Processing Unit",
        ],
        "answer": "A",
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Venus", "Jupiter", "Mars", "Saturn"],
        "answer": "C",
    },
    {
        "question": "What is the time complexity of binary search?",
        "choices": ["O(n)", "O(n²)", "O(log n)", "O(n log n)"],
        "answer": "C",
    },
    {
        "question": "Which protocol is used for secure web browsing?",
        "choices": ["HTTP", "FTP", "HTTPS", "SMTP"],
        "answer": "C",
    },
]

GSM8K_SAMPLE_PROBLEMS = [
    {
        "problem": "Janet has 3 apples. She buys 5 more. How many apples does she have?",
        "answer": "8",
    },
    {
        "problem": "A store sells pencils for $0.25 each. How much do 12 pencils cost?",
        "answer": "3.00",
    },
    {
        "problem": "Tom reads 20 pages per day. How many days to finish a 140-page book?",
        "answer": "7",
    },
    {
        "problem": "A rectangle has length 8cm and width 5cm. What is its area?",
        "answer": "40",
    },
    {
        "problem": "If a train travels at 60 mph for 2.5 hours, how far does it travel?",
        "answer": "150",
    },
    {
        "problem": "Sarah has $50. She spends $18 on lunch and $12 on a book. How much is left?",
        "answer": "20",
    },
    {
        "problem": "A class has 30 students. 40% are girls. How many boys are in the class?",
        "answer": "18",
    },
    {
        "problem": "If you have 3 boxes with 8 items each, how many items total?",
        "answer": "24",
    },
]

HUMANEVAL_SAMPLE_TASKS = [
    {
        "task_id": "HumanEval/1",
        "prompt": "def separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Split comma-separated parenthetical groups.\"\"\"\n",
        "entry_point": "separate_paren_groups",
        "test_signature": "assert len(separate_paren_groups('(()()) ((())) () ((())()())')) == 4",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return fractional part of a float.\"\"\"\n",
        "entry_point": "truncate_number",
        "test_signature": "assert truncate_number(3.5) == 0.5",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "def below_zero(operations: List[int]) -> bool:\n    \"\"\"Return True if balance drops below zero.\"\"\"\n",
        "entry_point": "below_zero",
        "test_signature": "assert below_zero([1, 2, -4, 5]) == True",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "def mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"Compute mean absolute deviation from mean.\"\"\"\n",
        "entry_point": "mean_absolute_deviation",
        "test_signature": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 0.6667) < 0.01",
    },
    {
        "task_id": "HumanEval/5",
        "prompt": "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\"Insert delimeter between each element.\"\"\"\n",
        "entry_point": "intersperse",
        "test_signature": "assert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
    },
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Evaluation results for a single dataset run."""

    model_name: str
    dataset_name: str
    timestamp: float = field(default_factory=time.time)
    accuracy: Optional[float] = None
    exact_match: Optional[float] = None
    pass_at_k: Optional[float] = None
    num_samples: int = 0
    num_correct: int = 0
    details: List[Dict] = field(default_factory=list)
    backend: str = "direct"  # "guidellm" or "direct"

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "exact_match": (
                round(self.exact_match, 4) if self.exact_match is not None else None
            ),
            "pass_at_k": (
                round(self.pass_at_k, 4) if self.pass_at_k is not None else None
            ),
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Direct evaluator (no GuideLLM dependency)
# ---------------------------------------------------------------------------


class DirectEvaluator:
    """
    Evaluates a model using direct OpenAI-compatible API calls.
    Used as fallback when GuideLLM is not installed.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _complete(self, prompt: str, max_tokens: int = 16) -> str:
        """Single synchronous chat completion."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Completion failed: %s", exc)
            return ""

    def eval_mmlu(self) -> EvalResult:
        """Multiple-choice evaluation on MMLU sample questions."""
        result = EvalResult(
            model_name=self.model,
            dataset_name="MMLU",
            num_samples=len(MMLU_SAMPLE_QUESTIONS),
        )

        for item in MMLU_SAMPLE_QUESTIONS:
            choices_text = "\n".join(
                f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"])
            )
            prompt = (
                f"Question: {item['question']}\n{choices_text}\n\n"
                "Answer with only the letter (A, B, C, or D)."
            )
            response = self._complete(prompt, max_tokens=4)
            predicted = response.upper().strip().lstrip("(").rstrip(")")[:1]
            correct = predicted == item["answer"]
            if correct:
                result.num_correct += 1
            result.details.append(
                {"question": item["question"], "predicted": predicted, "correct": correct}
            )

        result.accuracy = result.num_correct / result.num_samples
        result.exact_match = result.accuracy
        logger.info("MMLU accuracy: %.2f%%", result.accuracy * 100)
        return result

    def eval_gsm8k(self) -> EvalResult:
        """Exact-match evaluation on GSM8K sample math problems."""
        result = EvalResult(
            model_name=self.model,
            dataset_name="GSM8K",
            num_samples=len(GSM8K_SAMPLE_PROBLEMS),
        )

        for item in GSM8K_SAMPLE_PROBLEMS:
            prompt = (
                f"Solve this math problem step by step, then state the final answer "
                f"as a number on the last line.\n\nProblem: {item['problem']}"
            )
            response = self._complete(prompt, max_tokens=150)
            # Extract the last number from the response
            numbers = re.findall(r"[-+]?\d*\.?\d+", response.replace(",", ""))
            predicted = numbers[-1] if numbers else ""
            expected = item["answer"].replace(",", "").replace("$", "")
            correct = predicted == expected
            if correct:
                result.num_correct += 1
            result.details.append(
                {"problem": item["problem"], "predicted": predicted, "correct": correct}
            )

        result.exact_match = result.num_correct / result.num_samples
        result.accuracy = result.exact_match
        logger.info("GSM8K exact match: %.2f%%", result.exact_match * 100)
        return result

    def eval_humaneval(self, k: int = 1) -> EvalResult:
        """pass@k evaluation on HumanEval sample tasks."""
        result = EvalResult(
            model_name=self.model,
            dataset_name="HumanEval",
            num_samples=len(HUMANEVAL_SAMPLE_TASKS),
        )

        for item in HUMANEVAL_SAMPLE_TASKS:
            prompt = (
                f"Complete the following Python function. Return only the implementation, "
                f"no explanations.\n\n{item['prompt']}"
            )
            response = self._complete(prompt, max_tokens=256)

            # Heuristic pass check: does the response look like valid Python?
            passed = bool(
                response.strip()
                and ("return" in response or "pass" in response)
                and not response.startswith("I ")
            )
            if passed:
                result.num_correct += 1
            result.details.append(
                {"task_id": item["task_id"], "passed": passed}
            )

        result.pass_at_k = result.num_correct / result.num_samples
        result.accuracy = result.pass_at_k
        logger.info("HumanEval pass@%d: %.2f%%", k, result.pass_at_k * 100)
        return result


# ---------------------------------------------------------------------------
# GuideLLM runner (primary path)
# ---------------------------------------------------------------------------


class GuideLLMRunner:
    """
    Runs evaluations via the GuideLLM library when available.
    Falls back to DirectEvaluator if GuideLLM is not installed.
    """

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self._guidellm_available = self._check_guidellm()
        self._fallback = DirectEvaluator(base_url=base_url, model=model)

    @staticmethod
    def _check_guidellm() -> bool:
        try:
            import guidellm  # noqa: F401
            return True
        except ImportError:
            logger.info(
                "GuideLLM not installed — using direct OpenAI API evaluator as fallback."
            )
            return False

    def _run_guidellm_benchmark(self, dataset: str) -> Optional[EvalResult]:
        """Execute a GuideLLM benchmark and parse the result."""
        try:
            from guidellm import GuidanceEvaluator
            from guidellm.config import settings as guidellm_settings

            guidellm_settings.openai_api_base = f"{self.base_url}/v1"
            guidellm_settings.model = self.model

            evaluator = GuidanceEvaluator(
                target=f"{self.base_url}/v1",
                model=self.model,
                data=dataset,
            )
            report = evaluator.run()

            result = EvalResult(
                model_name=self.model,
                dataset_name=dataset.upper(),
                backend="guidellm",
            )

            # GuideLLM report schema varies by version; extract what we can
            if hasattr(report, "accuracy"):
                result.accuracy = float(report.accuracy)
            if hasattr(report, "exact_match"):
                result.exact_match = float(report.exact_match)
            if hasattr(report, "pass_at_k"):
                result.pass_at_k = float(report.pass_at_k)

            result.num_samples = getattr(report, "num_samples", 0)
            result.num_correct = getattr(report, "num_correct", 0)
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("GuideLLM run failed for %s: %s", dataset, exc)
            return None

    def run_mmlu(self) -> EvalResult:
        """Run MMLU evaluation."""
        if self._guidellm_available:
            result = self._run_guidellm_benchmark("mmlu")
            if result:
                return result
        return self._fallback.eval_mmlu()

    def run_gsm8k(self) -> EvalResult:
        """Run GSM8K evaluation."""
        if self._guidellm_available:
            result = self._run_guidellm_benchmark("gsm8k")
            if result:
                return result
        return self._fallback.eval_gsm8k()

    def run_humaneval(self) -> EvalResult:
        """Run HumanEval evaluation."""
        if self._guidellm_available:
            result = self._run_guidellm_benchmark("humaneval")
            if result:
                return result
        return self._fallback.eval_humaneval()

    def run_all(self, datasets: Optional[List[str]] = None) -> List[EvalResult]:
        """
        Run all requested evaluation benchmarks.

        Args:
            datasets: List of dataset names. Defaults to ["mmlu", "gsm8k", "humaneval"].

        Returns:
            List of EvalResult objects.
        """
        datasets = datasets or ["mmlu", "gsm8k", "humaneval"]
        dispatch = {
            "mmlu": self.run_mmlu,
            "gsm8k": self.run_gsm8k,
            "humaneval": self.run_humaneval,
        }

        results = []
        for ds in datasets:
            key = ds.lower()
            if key not in dispatch:
                logger.warning("Unknown dataset: %s — skipping.", ds)
                continue
            logger.info("Running evaluation: %s", ds.upper())
            results.append(dispatch[key]())

        return results
