import numpy as np
import pytest

from jev_crypto_research.council import (
    AggregationMethod,
    CouncilEngine,
    CouncilLaboratory,
    CouncilObservation,
    MultinomialLogisticStacker,
)
from jev_crypto_research.domain import ExpertDecision, Exposure
from jev_crypto_research.state import StateVariant


def test_all_eight_variants_have_255_nonempty_councils():
    assert len(CouncilEngine().subsets(list(StateVariant))) == 255


def test_aggregations_are_normalized():
    engine = CouncilEngine()
    preds = {
        StateVariant.V1_CANDLES: ExpertDecision(long=0.8, short=0.1, flat=0.1),
        StateVariant.V2_TECHNICAL: ExpertDecision(long=0.4, short=0.4, flat=0.2),
    }
    assert engine.mean_probability(preds).long == pytest.approx(0.6)
    assert engine.majority_vote(preds).long == 1.0
    weighted = engine.performance_weighted(
        preds, {StateVariant.V1_CANDLES: 3, StateVariant.V2_TECHNICAL: 1}
    )
    assert round(weighted.long, 3) == 0.7


def test_laboratory_scores_every_subset():
    predictions = {
        variant: ExpertDecision(long=0.7, short=0.2, flat=0.1)
        for variant in StateVariant
    }
    scorecard = CouncilLaboratory().evaluate_mean_subsets(
        [CouncilObservation(actual=Exposure.LONG, predictions=predictions)]
    )
    assert len(scorecard) == 255
    assert scorecard[(StateVariant.V1_CANDLES,)]["accuracy"] == 1.0


def test_laboratory_scores_all_three_methods_and_stacking_fits():
    predictions = {
        variant: ExpertDecision(long=0.7, short=0.2, flat=0.1)
        for variant in StateVariant
    }
    report = CouncilLaboratory().evaluate(
        [CouncilObservation(actual=Exposure.LONG, predictions=predictions)],
        performance_weights={variant: 1.0 for variant in StateVariant},
    )
    assert set(report) == set(AggregationMethod)
    assert all(len(scorecard) == 255 for scorecard in report.values())
    stacker = MultinomialLogisticStacker(epochs=20).fit(
        np.array([[1.0], [-1.0], [0.0]]), np.array([0, 1, 2])
    )
    assert stacker.predict_proba(np.array([[1.0]])).shape == (1, 3)
