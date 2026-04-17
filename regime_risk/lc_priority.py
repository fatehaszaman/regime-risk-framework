"""
lc_priority.py
--------------
LC priority allocation system.

Under normal conditions, USD capacity for import LCs is unconstrained.
During currency control regimes, central banks cap total USD allocations —
meaning a business must decide which LCs to prioritize when it can't open
all of them.

This module implements a priority allocation system that ranks pending LC
requests against available USD capacity, using a configurable scoring
model that weighs urgency, strategic importance, expiry risk, and cost.

The output is an allocation plan: which LCs to open now, which to defer,
and which to cancel, along with the reasoning for each decision.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LCStatus(str, Enum):
    APPROVED = "approved"       # Allocated — open this LC
    DEFERRED = "deferred"       # Capacity exhausted — defer to next cycle
    CANCELLED = "cancelled"     # Below minimum priority threshold — cancel


@dataclass
class PendingLC:
    """
    A pending LC request awaiting USD allocation.

    Parameters
    ----------
    lc_id : str
    commodity : str
    amount_usd : float
    urgency_score : float
        0–10. 10 = production stops without this shipment.
    strategic_score : float
        0–10. 10 = long-term supplier relationship at risk if skipped.
    days_until_expiry : int
        Days until the LC window closes. Lower = more urgent.
    unit_cost_usd : float
        Cost per unit. Used to assess cost efficiency of allocation.
    quantity : float
    unit : str
    notes : str
    """
    lc_id: str
    commodity: str
    amount_usd: float
    urgency_score: float            # 0–10
    strategic_score: float          # 0–10
    days_until_expiry: int
    unit_cost_usd: float
    quantity: float
    unit: str
    notes: str = ""


@dataclass
class AllocationDecision:
    """The allocation outcome for a single pending LC."""
    lc_id: str
    commodity: str
    amount_usd: float
    status: LCStatus
    priority_score: float
    reason: str


@dataclass
class AllocationPlan:
    """Full allocation plan for a given USD capacity."""
    total_capacity_usd: float
    total_allocated_usd: float
    total_deferred_usd: float
    total_cancelled_usd: float
    decisions: list[AllocationDecision]

    @property
    def utilization_rate(self) -> float:
        return self.total_allocated_usd / self.total_capacity_usd if self.total_capacity_usd > 0 else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "lc_id": d.lc_id,
                "commodity": d.commodity,
                "amount_usd": d.amount_usd,
                "status": d.status.value,
                "priority_score": d.priority_score,
                "reason": d.reason,
            }
            for d in self.decisions
        ])


class LCPriorityAllocator:
    """
    Allocates constrained USD capacity across pending LC requests.

    Priority scoring
    ----------------
    Each LC is scored on a 0–100 scale:

      priority = (
          urgency_weight    * urgency_score          # operational urgency
        + strategic_weight  * strategic_score        # relationship / strategic value
        + expiry_weight     * expiry_urgency          # time-to-expiry pressure
        + cost_weight       * cost_efficiency         # USD efficiency
      )

    where:
      expiry_urgency   = max(0, 10 - days_until_expiry / 3)  (peaks at 10 when expiry < 30d)
      cost_efficiency  = 10 * (1 - normalized_unit_cost)      (cheaper = higher score)

    Parameters
    ----------
    urgency_weight : float
    strategic_weight : float
    expiry_weight : float
    cost_weight : float
        Weights must sum to 1.0.
    min_priority_threshold : float
        LCs below this score are cancelled rather than deferred.
    """

    def __init__(
        self,
        urgency_weight: float = 0.40,
        strategic_weight: float = 0.25,
        expiry_weight: float = 0.20,
        cost_weight: float = 0.15,
        min_priority_threshold: float = 20.0,
    ):
        assert abs(urgency_weight + strategic_weight + expiry_weight + cost_weight - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.urgency_weight = urgency_weight
        self.strategic_weight = strategic_weight
        self.expiry_weight = expiry_weight
        self.cost_weight = cost_weight
        self.min_priority_threshold = min_priority_threshold

    def _expiry_urgency(self, days: int) -> float:
        """Higher score = closer to expiry."""
        return max(0.0, 10.0 - days / 3.0)

    def _cost_efficiency(self, unit_cost: float, max_cost: float) -> float:
        """Higher score = lower unit cost relative to peers."""
        if max_cost == 0:
            return 5.0
        return 10.0 * (1.0 - unit_cost / max_cost)

    def _priority_score(self, lc: PendingLC, max_unit_cost: float) -> float:
        expiry_urgency = self._expiry_urgency(lc.days_until_expiry)
        cost_eff = self._cost_efficiency(lc.unit_cost_usd, max_unit_cost)
        score = (
            self.urgency_weight   * lc.urgency_score  * 10
            + self.strategic_weight * lc.strategic_score * 10
            + self.expiry_weight    * expiry_urgency   * 10
            + self.cost_weight      * cost_eff         * 10
        )
        return round(score, 2)

    def allocate(
        self,
        pending_lcs: list[PendingLC],
        available_usd: float,
    ) -> AllocationPlan:
        """
        Allocate available USD capacity across pending LCs.

        LCs are ranked by priority score. Highest-priority LCs are
        approved first until capacity is exhausted. Remaining LCs
        are deferred or cancelled based on their score.

        Parameters
        ----------
        pending_lcs : list[PendingLC]
        available_usd : float
            Total USD capacity available for LC allocation this cycle.

        Returns
        -------
        AllocationPlan
        """
        if not pending_lcs:
            return AllocationPlan(available_usd, 0, 0, 0, [])

        max_unit_cost = max(lc.unit_cost_usd for lc in pending_lcs)

        scored = sorted(
            [(lc, self._priority_score(lc, max_unit_cost)) for lc in pending_lcs],
            key=lambda x: x[1],
            reverse=True,
        )

        remaining = available_usd
        decisions = []
        allocated = deferred = cancelled = 0.0

        for lc, score in scored:
            if remaining >= lc.amount_usd:
                status = LCStatus.APPROVED
                reason = f"Priority {score:.1f}/100 — allocated within capacity"
                remaining -= lc.amount_usd
                allocated += lc.amount_usd
            elif score < self.min_priority_threshold:
                status = LCStatus.CANCELLED
                reason = f"Priority {score:.1f}/100 — below minimum threshold ({self.min_priority_threshold})"
                cancelled += lc.amount_usd
            else:
                status = LCStatus.DEFERRED
                reason = f"Priority {score:.1f}/100 — capacity exhausted, defer to next cycle"
                deferred += lc.amount_usd

            decisions.append(AllocationDecision(
                lc_id=lc.lc_id,
                commodity=lc.commodity,
                amount_usd=lc.amount_usd,
                status=status,
                priority_score=score,
                reason=reason,
            ))

        return AllocationPlan(
            total_capacity_usd=available_usd,
            total_allocated_usd=round(allocated, 2),
            total_deferred_usd=round(deferred, 2),
            total_cancelled_usd=round(cancelled, 2),
            decisions=decisions,
        )
