# protomind/task/reframe_generator.py

from __future__ import annotations

from dataclasses import dataclass
    task_name: str
    failed_component: str
    current_observation: Any
    recent_observations: List[Any]
    environment_immutable: bool
    tools_exhausted: bool
    last_action: Optional[Any] = None
    tool_name: Optional[str] = None
    tool_affordances: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
class ReframeContext:
    """
    Minimal evidence bundle (data only).
    Keep this dumb and explicit to avoid reward/goal leakage.
    """
    task_name: str
    failed_component: str  # e.g. "tool", "environment", "representation", "planner"
    current_observation: Any
    recent_observations: List[Any]

    environment_immutable: bool  # result of probe (or equivalent)
    tools_exhausted: bool        # runtime says: no remaining tool variants to try

    # Optional extras for explainability only (never used as reward)
    last_action: Optional[Any] = None
    tool_name: Optional[str] = None
    tool_affordances: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ReframeGenerator:
    """
    Produces a single ReframeProposal from evidence.

    IMPORTANT:
    - This does NOT execute anything.
    - This does NOT search multiple reframes.
    - This does NOT soften the goal.
    """

    def generate(self, ctx: ReframeContext) -> Optional[ReframeProposal]:
        # If we don't have evidence of stall, don't reframe.
        if not ctx.recent_observations:
            return None

        # If probe says environment can't change, start with Level 1 (representation)
        # because it's the least invasive (and least gameable).
        if ctx.environment_immutable:
            return self._level1_representation(ctx)

        # If environment *can* change but we're stuck, that’s usually a missing capability/tool class.
        if ctx.tools_exhausted:
            return self._level2_capability(ctx)

        # Otherwise, we’re stuck but not clearly immutable and not tool-exhausted:
        # propose a Level 1 representation tweak first.
        return self._level1_representation(ctx)

    # -----------------------------
    # Level 1: representation change
    # -----------------------------
    def _level1_representation(self, ctx: ReframeContext) -> ReframeProposal:
        # Conservative “truth-preserving” representation changes.
        # These do NOT alter the goal; they change how we *describe/track* progress.
            rep = self._choose_representation_change(ctx)

        proposal = ReframeProposal(
            level=1,
            reason=(
                "Stagnation detected. Proposing a representation change to avoid "
                "repeating identical observations without introducing new difficulty assumptions."
            ),
            failed_component=ctx.failed_component,
            representation_change=rep,
            preserves_goal=True,
            preserves_environment_truth=True,
            evidence=self._evidence(ctx),
        )

        # Runtime must reject invalid proposals.
        return proposal if proposal.is_valid() else None

    def _choose_representation_change(self, ctx: ReframeContext) -> str:
        obs0 = ctx.recent_observations[0]
        # A few generic but useful transforms:
        # - delta encoding helps when absolute state is constant but transitions matter
        # - key-field projection reduces noise
        # - temporal windowing can turn repeated frames into "no-op" signal
        if isinstance(obs0, dict):
            keys = list(obs0.keys())
            # prefer stable, readable changes
            if "pos" in keys:
                return "Switch state representation from absolute 'pos' to Δpos per step and detect no-op actions explicitly."
            return f"Project observation dict to a stable key subset {keys[:3]} (or a canonical ordering) to avoid noisy equality checks."
        return "Switch to a delta-based or tokenized representation of observations (e.g., hash/normalize), tracking explicit 'no-change' transitions."

    # -----------------------------
    # Level 2: capability needed
    # -----------------------------
    def _level2_capability(self, ctx: ReframeContext) -> ReframeProposal:
        cap = self._infer_capability_needed(ctx)

        proposal = ReframeProposal(
            level=2,
            reason=(
                "Environment appears changeable, but tool variants are exhausted under current affordances. "
                "Proposing a new capability/tool-class requirement without changing the goal."
            ),
            failed_component=ctx.failed_component,
            capability_needed=cap,
            preserves_goal=True,
            preserves_environment_truth=True,
            evidence=self._evidence(ctx),
        )

        return proposal if proposal.is_valid() else None

    def _infer_capability_needed(self, ctx: ReframeContext) -> str:
        # Keep this brutally generic. Do NOT bake in “how to win”.
        # We're just naming missing *types* of interaction.
        if isinstance(ctx.current_observation, dict):
            if "pos" in ctx.current_observation:
                return "A tool class that can modify position via a different mechanism than current 'move' (e.g., jump/teleport/push/pull), OR a tool that changes environment constraints."
        return "A new tool class that can change the environment state in a way current affordances cannot (state-altering, constraint-altering, or access-expanding)."

    # -----------------------------
    # Level 3: assumption shift (dangerous)
    # -----------------------------
    def level3_assumption_shift(self, ctx: ReframeContext, assumption_shift: str) -> Optional[ReframeProposal]:
        """
        Level 3 is *manual* / explicit. You call it only when you KNOW why.
        This prevents the system from auto-generating self-serving stories.

        Example assumption_shift:
          "The environment may require a prerequisite state not reachable with current toolset."
        """
        proposal = ReframeProposal(
            level=3,
            reason=(
                "Danger-level reframe: assumption shift proposed explicitly. "
                "Must preserve environment truth and must not soften the goal."
            ),
            failed_component=ctx.failed_component,
            assumption_shift=assumption_shift,
            preserves_goal=True,
            preserves_environment_truth=True,
            evidence=self._evidence(ctx),
        )
        return proposal if proposal.is_valid() else None

    # -----------------------------
    # Evidence capture
    # -----------------------------
    def _evidence(self, ctx: ReframeContext) -> Dict[str, Any]:
        # Evidence is for logging/human inspection only.
        # It must not be used to score or select winners.
        return {
            "task_name": ctx.task_name,
            "failed_component": ctx.failed_component,
            "environment_immutable": ctx.environment_immutable,
            "tools_exhausted": ctx.tools_exhausted,
            "tool_name": ctx.tool_name,
            "tool_affordances": ctx.tool_affordances,
            "last_action": repr(ctx.last_action),
            "current_observation_type": type(ctx.current_observation).__name__,
            "recent_observations_count": len(ctx.recent_observations),
            "notes": ctx.notes,
        }
