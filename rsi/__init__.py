"""Kayak 10.9 Recursive Self-Improvement evaluation harness and domain models."""

from __future__ import annotations

from rsi.baseline import (
    BaselineReference,
    CapytaineBaseline,
    FairnessBaseline,
    FairnessCutBaseline,
    FairnessSectionBaseline,
    LoadBaseline,
    OpenFoamBaseline,
    OpenFoamSpeedBaseline,
    SeakeepingConditionBaseline,
    load_baseline_reference,
)
from rsi.diagnose import (
    AdaptationHints,
    DiagnosticItem,
    DiagnosticReport,
    aggregate_diagnostics,
    diagnose_candidate,
    diagnose_validation_error,
)
from rsi.evaluator import CandidateMetrics, LoadEvalMetrics, evaluate_tier0_tier1
from rsi.fidelity import (
    CfdMetrics,
    SeakeepingMetrics,
    evaluate_tier2_cfd,
    evaluate_tier2_seakeeping,
    parse_openfoam_case_metrics,
    resolve_openfoam_bashrc,
    stage_openfoam_case,
)
from rsi.gates import GateReport, GateViolation, evaluate_gates
from rsi.genome import DesignSpace, ParameterSpec, hull_to_vector, vector_to_hull
from rsi.loop import (
    CampaignConfig,
    CampaignResult,
    CampaignState,
    LedgerEntry,
    ResumeStateError,
    load_campaign_status,
    run_campaign,
    run_smoke,
)
from rsi.promote import (
    ComparisonReport,
    PromotionReport,
    compare_candidate,
    promote_candidate,
)

__all__ = [
    "AdaptationHints",
    "BaselineReference",
    "CampaignConfig",
    "CampaignResult",
    "CampaignState",
    "CandidateMetrics",
    "CapytaineBaseline",
    "CfdMetrics",
    "ComparisonReport",
    "DesignSpace",
    "DiagnosticItem",
    "DiagnosticReport",
    "FairnessBaseline",
    "FairnessCutBaseline",
    "FairnessSectionBaseline",
    "GateReport",
    "GateViolation",
    "LedgerEntry",
    "LoadBaseline",
    "LoadEvalMetrics",
    "OpenFoamBaseline",
    "OpenFoamSpeedBaseline",
    "ParameterSpec",
    "PromotionReport",
    "ResumeStateError",
    "SeakeepingConditionBaseline",
    "SeakeepingMetrics",
    "aggregate_diagnostics",
    "compare_candidate",
    "diagnose_candidate",
    "diagnose_validation_error",
    "evaluate_gates",
    "evaluate_tier0_tier1",
    "evaluate_tier2_cfd",
    "evaluate_tier2_seakeeping",
    "hull_to_vector",
    "load_baseline_reference",
    "load_campaign_status",
    "promote_candidate",
    "resolve_openfoam_bashrc",
    "run_campaign",
    "run_smoke",
    "stage_openfoam_case",
    "vector_to_hull",
    "parse_openfoam_case_metrics",
]
