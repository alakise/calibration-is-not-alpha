from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np

from .domain import ExpertDecision, Exposure
from .state import StateVariant


class CouncilEngine:
    def subsets(self, variants: list[StateVariant]) -> list[tuple[StateVariant, ...]]:
        """All 255 non-empty sets when passed all eight state variants."""
        return [
            item
            for size in range(1, len(variants) + 1)
            for item in combinations(variants, size)
        ]

    def mean_probability(
        self, predictions: Mapping[StateVariant, ExpertDecision]
    ) -> ExpertDecision:
        probs = list(predictions.values())
        return ExpertDecision(
            long=np.mean([p.long for p in probs]),
            short=np.mean([p.short for p in probs]),
            flat=np.mean([p.flat for p in probs]),
        ).normalized()

    def majority_vote(
        self, predictions: Mapping[StateVariant, ExpertDecision]
    ) -> ExpertDecision:
        votes = {
            direction: sum(p.winner() is direction for p in predictions.values())
            for direction in Exposure
        }
        # Retain each voter's certainty as a deterministic tie-breaker.
        certainty = {
            direction: sum(getattr(p, direction.value) for p in predictions.values())
            for direction in Exposure
        }
        chosen = max(Exposure, key=lambda d: (votes[d], certainty[d]))
        return ExpertDecision(
            long=float(chosen is Exposure.LONG),
            short=float(chosen is Exposure.SHORT),
            flat=float(chosen is Exposure.FLAT),
        )

    def performance_weighted(
        self,
        predictions: Mapping[StateVariant, ExpertDecision],
        weights: Mapping[StateVariant, float],
    ) -> ExpertDecision:
        if set(predictions) - set(weights):
            raise ValueError(
                "each council member needs an out-of-sample performance weight"
            )
        total = sum(max(0.0, weights[v]) for v in predictions)
        if total <= 0:
            raise ValueError("at least one performance weight must be positive")
        return ExpertDecision(
            long=sum(predictions[v].long * max(0.0, weights[v]) for v in predictions)
            / total,
            short=sum(predictions[v].short * max(0.0, weights[v]) for v in predictions)
            / total,
            flat=sum(predictions[v].flat * max(0.0, weights[v]) for v in predictions)
            / total,
        ).normalized()


class AggregationMethod(StrEnum):
    MEAN = "mean_probability"
    MAJORITY = "majority_vote"
    PERFORMANCE_WEIGHTED = "performance_weighted"


class MultinomialLogisticStacker:
    """Small dependency-free multinomial stacker. Train only on a prior time slice."""

    def __init__(
        self, learning_rate: float = 0.05, epochs: int = 500, l2: float = 0.001
    ):
        self.learning_rate, self.epochs, self.l2 = learning_rate, epochs, l2
        self._weights: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, labels: np.ndarray
    ) -> "MultinomialLogisticStacker":
        if features.ndim != 2 or len(features) != len(labels) or len(features) < 2:
            raise ValueError("features and labels need at least two aligned rows")
        if not set(labels).issubset({0, 1, 2}):
            raise ValueError("labels must be 0=long, 1=short, 2=flat")
        x = np.column_stack((np.ones(len(features)), features))
        weights = np.zeros((x.shape[1], 3))
        target = np.eye(3)[labels]
        for _ in range(self.epochs):
            scores = x @ weights
            exps = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            gradient = x.T @ (probs - target) / len(x)
            gradient[1:] += self.l2 * weights[1:]
            weights -= self.learning_rate * gradient
        self._weights = weights
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("fit the stacker on a prior training period first")
        x = np.column_stack((np.ones(len(features)), features))
        scores = x @ self._weights
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class CouncilObservation:
    actual: Exposure
    predictions: Mapping[StateVariant, ExpertDecision]


class CouncilLaboratory:
    """Offline scorer. Input rows must be strictly chronological out-of-sample decisions."""

    def __init__(self, engine: CouncilEngine | None = None):
        self.engine = engine or CouncilEngine()

    def evaluate(
        self,
        observations: Sequence[CouncilObservation],
        methods: Sequence[AggregationMethod] = tuple(AggregationMethod),
        performance_weights: Mapping[StateVariant, float] | None = None,
    ) -> dict[AggregationMethod, dict[tuple[StateVariant, ...], dict[str, float]]]:
        if not observations:
            return {}
        variants = list(observations[0].predictions)
        if (
            AggregationMethod.PERFORMANCE_WEIGHTED in methods
            and performance_weights is None
        ):
            raise ValueError(
                "performance_weights must come from a prior out-of-sample calibration period"
            )
        results: dict[
            AggregationMethod, dict[tuple[StateVariant, ...], dict[str, float]]
        ] = {}
        for method in methods:
            scorecard = {}
            for subset in self.engine.subsets(variants):
                probabilities = [
                    self._aggregate(
                        method,
                        {variant: row.predictions[variant] for variant in subset},
                        performance_weights,
                    )
                    for row in observations
                ]
                accuracy = sum(
                    p.winner() is row.actual
                    for p, row in zip(probabilities, observations, strict=True)
                ) / len(observations)
                brier = sum(
                    (getattr(p, direction.value) - float(row.actual is direction)) ** 2
                    for p, row in zip(probabilities, observations, strict=True)
                    for direction in Exposure
                ) / len(observations)
                scorecard[subset] = {"accuracy": accuracy, "multiclass_brier": brier}
            results[method] = scorecard
        return results

    def evaluate_mean_subsets(
        self, observations: list[CouncilObservation]
    ) -> dict[tuple[StateVariant, ...], dict[str, float]]:
        return self.evaluate(observations, methods=(AggregationMethod.MEAN,))[
            AggregationMethod.MEAN
        ]

    def _aggregate(
        self,
        method: AggregationMethod,
        predictions: Mapping[StateVariant, ExpertDecision],
        weights: Mapping[StateVariant, float] | None,
    ) -> ExpertDecision:
        if method is AggregationMethod.MEAN:
            return self.engine.mean_probability(predictions)
        if method is AggregationMethod.MAJORITY:
            return self.engine.majority_vote(predictions)
        assert weights is not None
        return self.engine.performance_weighted(predictions, weights)
