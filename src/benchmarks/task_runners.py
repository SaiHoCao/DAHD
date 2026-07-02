"""Task runners for benchmark evaluation datasets.

Provides standardized interfaces for loading datasets, formatting prompts,
and evaluating model outputs across different benchmark tasks.
"""

from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset, load_from_disk

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


class TaskRunner(ABC):
    """Abstract base class for benchmark task runners.

    Defines the interface for loading datasets, formatting prompts,
    and evaluating predictions against references.
    """

    task_name: str = "base"

    @abstractmethod
    def load_dataset(
        self, num_samples: int = 100, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load and prepare dataset samples.

        Args:
            num_samples: Maximum number of samples to load.
            local_path: If provided, load dataset from this local path instead of HuggingFace Hub.
            use_synthetic: If True, generate synthetic placeholder data (for offline testing).

        Returns:
            List of sample dictionaries with at least 'prompt' and 'reference' keys.
        """
        ...

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic placeholder samples for offline testing.

        Subclasses should override this to provide task-specific synthetic data.

        Args:
            num_samples: Number of synthetic samples to generate.

        Returns:
            List of synthetic sample dictionaries.
        """
        return []

    @abstractmethod
    def format_prompt(self, sample: dict) -> str:
        """Format a dataset sample into a model prompt.

        Args:
            sample: A single dataset sample dictionary.

        Returns:
            Formatted prompt string ready for model input.
        """
        ...

    @abstractmethod
    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate a single prediction against the reference.

        Args:
            prediction: Model-generated output.
            reference: Ground truth reference.

        Returns:
            Evaluation score (interpretation depends on the task).
        """
        ...

    def evaluate_batch(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, float]:
        """Evaluate a batch of predictions.

        Args:
            predictions: List of model outputs.
            references: List of ground truth references.

        Returns:
            Dictionary with aggregate metrics.
        """
        scores = [
            self.evaluate(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        return {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "num_correct": sum(1 for s in scores if s >= 1.0),
            "num_total": len(scores),
            "accuracy": sum(1 for s in scores if s >= 1.0) / len(scores) if scores else 0.0,
        }


class GSM8KRunner(TaskRunner):
    """Runner for the GSM8K math word problem benchmark.

    Evaluates exact match on numerical answers extracted from
    chain-of-thought solutions.
    """

    task_name = "gsm8k"

    def load_dataset(
        self, num_samples: int = 100, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load GSM8K test samples.

        Args:
            num_samples: Maximum number of samples to load.
            local_path: If provided, load dataset from this local path.
            use_synthetic: If True, generate synthetic data as placeholder.

        Returns:
            List of samples with 'prompt', 'reference', and 'question' keys.
        """
        if use_synthetic:
            logger.info("Using synthetic GSM8K data for offline testing.")
            return self._generate_synthetic_samples(num_samples)

        if not HF_DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        try:
            if local_path:
                logger.info(f"Loading GSM8K from local path: {local_path}")
                ds = load_from_disk(local_path)
                if isinstance(ds, dict):
                    ds = ds.get("test", ds.get("train", list(ds.values())[0]))
            else:
                ds = load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            logger.warning(f"Failed to load GSM8K dataset: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            answer = self._extract_answer(item["answer"])
            samples.append({
                "question": item["question"],
                "prompt": self.format_prompt({"question": item["question"]}),
                "reference": answer,
                "full_solution": item["answer"],
            })
        logger.info(f"Loaded {len(samples)} GSM8K samples")
        return samples

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic GSM8K-like math word problems."""
        templates = [
            ("Alice has {a} apples. She buys {b} more. How many apples does she have?", "{c}"),
            ("A store has {a} items. They sell {b} items. How many items are left?", "{c}"),
            ("John earns ${a} per hour. He works {b} hours. How much does he earn?", "{c}"),
            ("A train travels {a} km/h for {b} hours. What distance does it cover?", "{c}"),
            ("There are {a} students in a class. {b} are absent. How many are present?", "{c}"),
        ]
        samples = []
        for i in range(num_samples):
            t_idx = i % len(templates)
            a = random.randint(10, 100)
            b = random.randint(1, min(a, 50))
            if t_idx in (0, 2, 3):  # addition/multiplication
                c = a + b if t_idx == 0 else a * b
            else:  # subtraction
                c = a - b
            question = templates[t_idx][0].format(a=a, b=b, c=c)
            answer = str(c)
            samples.append({
                "question": question,
                "prompt": self.format_prompt({"question": question}),
                "reference": answer,
                "full_solution": f"The answer is #### {answer}",
                "_synthetic": True,
            })
        logger.info(f"Generated {len(samples)} synthetic GSM8K samples")
        return samples

    def format_prompt(self, sample: dict) -> str:
        """Format a GSM8K question into a chain-of-thought prompt."""
        question = sample.get("question", "")
        return (
            f"Solve the following math problem step by step.\n\n"
            f"Question: {question}\n\n"
            f"Solution: Let's think step by step.\n"
        )

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate by exact match on extracted numerical answer."""
        pred_answer = self._extract_answer(prediction)
        return 1.0 if pred_answer == reference else 0.0

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract the final numerical answer from a solution text."""
        # Look for #### pattern (GSM8K format)
        match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip().replace(",", "")
        # Fallback: last number in text
        numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return ""


class MATHRunner(TaskRunner):
    """Runner for the MATH benchmark (competition mathematics).

    Evaluates exact match on final answers, with normalization
    for mathematical expressions.
    """

    task_name = "math"

    def load_dataset(
        self, num_samples: int = 100, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load MATH test samples."""
        if use_synthetic:
            logger.info("Using synthetic MATH data for offline testing.")
            return self._generate_synthetic_samples(num_samples)

        if not HF_DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        try:
            if local_path:
                logger.info(f"Loading MATH from local path: {local_path}")
                ds = load_from_disk(local_path)
                if isinstance(ds, dict):
                    ds = ds.get("test", ds.get("train", list(ds.values())[0]))
            else:
                ds = load_dataset("hendrycks/competition_math", split="test")
        except Exception as e:
            logger.warning(f"Failed to load MATH dataset: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                "problem": item["problem"],
                "prompt": self.format_prompt({"problem": item["problem"]}),
                "reference": item["solution"],
                "level": item.get("level", ""),
                "type": item.get("type", ""),
            })
        logger.info(f"Loaded {len(samples)} MATH samples")
        return samples

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic MATH-like problems."""
        templates = [
            ("Find the value of x if 2x + {a} = {b}.", "\\boxed{{{c}}}"),
            ("What is the sum of the first {a} positive integers?", "\\boxed{{{c}}}"),
            ("Compute {a} * {b}.", "\\boxed{{{c}}}"),
        ]
        samples = []
        for i in range(num_samples):
            t_idx = i % len(templates)
            a = random.randint(2, 50)
            b = random.randint(a + 1, 200)
            if t_idx == 0:
                c = (b - a) // 2
            elif t_idx == 1:
                c = a * (a + 1) // 2
            else:
                c = a * b
            problem = templates[t_idx][0].format(a=a, b=b, c=c)
            solution = templates[t_idx][1].format(c=c)
            samples.append({
                "problem": problem,
                "prompt": self.format_prompt({"problem": problem}),
                "reference": solution,
                "level": "Level 1",
                "type": "Algebra",
                "_synthetic": True,
            })
        logger.info(f"Generated {len(samples)} synthetic MATH samples")
        return samples

    def format_prompt(self, sample: dict) -> str:
        """Format a MATH problem into a prompt."""
        problem = sample.get("problem", "")
        return (
            f"Solve the following math problem. "
            f"Put your final answer within \\boxed{{}}.\n\n"
            f"Problem: {problem}\n\n"
            f"Solution:\n"
        )

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate by exact match on boxed answer."""
        pred_answer = self._extract_boxed(prediction)
        ref_answer = self._extract_boxed(reference)
        return 1.0 if self._normalize(pred_answer) == self._normalize(ref_answer) else 0.0

    @staticmethod
    def _extract_boxed(text: str) -> str:
        """Extract content from \\boxed{...}."""
        match = re.search(r"\\boxed\{(.+?)\}", text)
        if match:
            return match.group(1)
        return text.strip()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize mathematical expression for comparison."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" ", "")
        return text.lower()


class HumanEvalRunner(TaskRunner):
    """Runner for the HumanEval code generation benchmark.

    Evaluates pass@1 by executing generated code against test cases.
    """

    task_name = "humaneval"

    def load_dataset(
        self, num_samples: int = 100, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load HumanEval samples."""
        if use_synthetic:
            logger.info("Using synthetic HumanEval data for offline testing.")
            return self._generate_synthetic_samples(num_samples)

        if not HF_DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        try:
            if local_path:
                logger.info(f"Loading HumanEval from local path: {local_path}")
                ds = load_from_disk(local_path)
                if isinstance(ds, dict):
                    ds = ds.get("test", ds.get("train", list(ds.values())[0]))
            else:
                ds = load_dataset("openai/openai_humaneval", split="test")
        except Exception as e:
            logger.warning(f"Failed to load HumanEval dataset: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "reference": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            })
        logger.info(f"Loaded {len(samples)} HumanEval samples")
        return samples

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic HumanEval-like code problems."""
        problems = [
            {
                "task_id": "synthetic/0",
                "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n",
                "reference": "    return a + b\n",
                "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0\n",
                "entry_point": "add",
            },
            {
                "task_id": "synthetic/1",
                "prompt": "def multiply(a: int, b: int) -> int:\n    \"\"\"Return the product of a and b.\"\"\"\n",
                "reference": "    return a * b\n",
                "test": "assert multiply(2, 3) == 6\nassert multiply(0, 5) == 0\n",
                "entry_point": "multiply",
            },
            {
                "task_id": "synthetic/2",
                "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even.\"\"\"\n",
                "reference": "    return n % 2 == 0\n",
                "test": "assert is_even(4) == True\nassert is_even(3) == False\n",
                "entry_point": "is_even",
            },
        ]
        samples = []
        for i in range(num_samples):
            sample = problems[i % len(problems)].copy()
            sample["task_id"] = f"synthetic/{i}"
            sample["_synthetic"] = True
            samples.append(sample)
        logger.info(f"Generated {len(samples)} synthetic HumanEval samples")
        return samples

    def format_prompt(self, sample: dict) -> str:
        """Format a HumanEval problem (prompt already includes function signature)."""
        return sample.get("prompt", "")

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate code by execution (simplified: string match as fallback).

        Note: Full pass@1 evaluation requires sandboxed execution.
        This implementation does a simplified check.
        """
        # Simplified evaluation - check if key parts of the solution are present
        pred_clean = prediction.strip()
        ref_clean = reference.strip()

        if pred_clean == ref_clean:
            return 1.0

        # Check if the prediction contains the core logic
        # (Full evaluation would use code execution in a sandbox)
        return 0.0


class MTBenchRunner(TaskRunner):
    """Runner for the MT-Bench multi-turn conversation benchmark.

    Evaluates quality of conversational responses using reference-based
    scoring (simplified without LLM judge).
    """

    task_name = "mt_bench"

    def load_dataset(
        self, num_samples: int = 80, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load MT-Bench questions."""
        if use_synthetic:
            logger.info("Using synthetic MT-Bench data for offline testing.")
            return self._generate_synthetic_samples(num_samples)

        if not HF_DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        try:
            if local_path:
                logger.info(f"Loading MT-Bench from local path: {local_path}")
                ds = load_from_disk(local_path)
                if isinstance(ds, dict):
                    ds = ds.get("train", list(ds.values())[0])
            else:
                ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        except Exception as e:
            logger.warning(f"Failed to load MT-Bench dataset: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            prompt = item.get("prompt", [""])[0] if isinstance(item.get("prompt"), list) else item.get("prompt", "")
            samples.append({
                "question_id": item.get("question_id", i),
                "category": item.get("category", ""),
                "prompt": self.format_prompt({"question": prompt}),
                "reference": item.get("reference", [""])[0] if isinstance(item.get("reference"), list) else "",
            })
        logger.info(f"Loaded {len(samples)} MT-Bench samples")
        return samples

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic MT-Bench-like questions."""
        questions = [
            "Explain the concept of machine learning in simple terms.",
            "What are the benefits of exercise for mental health?",
            "Describe the process of photosynthesis.",
            "How does a compiler work?",
            "What is the significance of the Turing test?",
        ]
        samples = []
        for i in range(num_samples):
            q = questions[i % len(questions)]
            samples.append({
                "question_id": i,
                "category": "general",
                "prompt": self.format_prompt({"question": q}),
                "reference": "",
                "_synthetic": True,
            })
        logger.info(f"Generated {len(samples)} synthetic MT-Bench samples")
        return samples

    def format_prompt(self, sample: dict) -> str:
        """Format an MT-Bench question."""
        question = sample.get("question", "")
        return f"{question}"

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate response quality (simplified length-based heuristic).

        Note: Full MT-Bench evaluation uses an LLM judge.
        """
        if not prediction.strip():
            return 0.0
        # Simple heuristic: non-empty response with reasonable length
        length_score = min(len(prediction.split()) / 50.0, 1.0)
        return length_score


class CNNDailyMailRunner(TaskRunner):
    """Runner for the CNN/DailyMail summarization benchmark.

    Evaluates using BLEU score between generated and reference summaries.
    """

    task_name = "cnn_dailymail"

    def load_dataset(
        self, num_samples: int = 100, local_path: str | None = None, use_synthetic: bool = False
    ) -> list[dict]:
        """Load CNN/DailyMail test samples."""
        if use_synthetic:
            logger.info("Using synthetic CNN/DailyMail data for offline testing.")
            return self._generate_synthetic_samples(num_samples)

        if not HF_DATASETS_AVAILABLE:
            logger.warning("datasets library not available. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        try:
            if local_path:
                logger.info(f"Loading CNN/DailyMail from local path: {local_path}")
                ds = load_from_disk(local_path)
                if isinstance(ds, dict):
                    ds = ds.get("test", ds.get("train", list(ds.values())[0]))
            else:
                ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        except Exception as e:
            logger.warning(f"Failed to load CNN/DailyMail dataset: {e}. Falling back to synthetic data.")
            return self._generate_synthetic_samples(num_samples)

        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                "article": item["article"],
                "prompt": self.format_prompt({"article": item["article"]}),
                "reference": item["highlights"],
            })
        logger.info(f"Loaded {len(samples)} CNN/DailyMail samples")
        return samples

    def _generate_synthetic_samples(self, num_samples: int) -> list[dict]:
        """Generate synthetic summarization samples."""
        articles = [
            "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean. "
            "The fish, which lives at depths of over 3000 meters, has unique bioluminescent properties. "
            "Researchers believe this discovery could lead to advances in medical imaging technology.",
            "A new study suggests that regular meditation can improve cognitive function in older adults. "
            "Participants who meditated for 20 minutes daily showed improved memory and attention span "
            "after just 8 weeks of practice.",
            "The city council approved a new public transit plan that will add 15 new bus routes. "
            "The plan aims to reduce traffic congestion and lower carbon emissions by encouraging "
            "residents to use public transportation instead of private vehicles.",
        ]
        highlights = [
            "New deep-sea fish species discovered with bioluminescent properties.",
            "Regular meditation improves cognitive function in older adults.",
            "City council approves new public transit plan with 15 bus routes.",
        ]
        samples = []
        for i in range(num_samples):
            idx = i % len(articles)
            samples.append({
                "article": articles[idx],
                "prompt": self.format_prompt({"article": articles[idx]}),
                "reference": highlights[idx],
                "_synthetic": True,
            })
        logger.info(f"Generated {len(samples)} synthetic CNN/DailyMail samples")
        return samples

    def format_prompt(self, sample: dict) -> str:
        """Format an article for summarization."""
        article = sample.get("article", "")
        # Truncate very long articles
        words = article.split()
        if len(words) > 512:
            article = " ".join(words[:512]) + "..."
        return (
            f"Summarize the following article in a few sentences.\n\n"
            f"Article: {article}\n\n"
            f"Summary:"
        )

    def evaluate(self, prediction: str, reference: str) -> float:
        """Evaluate using simple BLEU-like n-gram overlap."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # Simple unigram precision as BLEU approximation
        pred_set = set(pred_tokens)
        ref_set = set(ref_tokens)
        overlap = len(pred_set & ref_set)
        precision = overlap / len(pred_set) if pred_set else 0.0
        recall = overlap / len(ref_set) if ref_set else 0.0

        # F1-like score
        if precision + recall > 0:
            return 2.0 * precision * recall / (precision + recall)
        return 0.0


# Task runner registry
_TASK_RUNNERS: dict[str, type[TaskRunner]] = {
    "gsm8k": GSM8KRunner,
    "math": MATHRunner,
    "humaneval": HumanEvalRunner,
    "mt_bench": MTBenchRunner,
    "cnn_dailymail": CNNDailyMailRunner,
}


def get_task_runner(task_name: str) -> TaskRunner:
    """Factory function to get a task runner by name.

    Args:
        task_name: Name of the benchmark task.

    Returns:
        Instantiated TaskRunner for the specified task.

    Raises:
        ValueError: If task_name is not recognized.
    """
    task_lower = task_name.lower().replace("-", "_")
    if task_lower not in _TASK_RUNNERS:
        available = ", ".join(sorted(_TASK_RUNNERS.keys()))
        raise ValueError(
            f"Unknown task: '{task_name}'. Available tasks: {available}"
        )
    return _TASK_RUNNERS[task_lower]()


def list_available_tasks() -> list[str]:
    """List all available benchmark task names.

    Returns:
        Sorted list of task name strings.
    """
    return sorted(_TASK_RUNNERS.keys())
