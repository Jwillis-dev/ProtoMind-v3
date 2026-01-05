# Changelog (session snapshot)

Generated: 2026-01-05 15:52 UTC

## Added
- task/__init__.py (task is now a real package)
- environment/__init__.py
- task/reframe_proposal.py (data-only reframe proposal)
- task/reframe_generator.py (one-shot proposal generator)
- task/reframe_apply.py (Level-1 mechanical observation transforms)
- task/episode_state.py (one-shot reframe fuse + budgets + de-dup)
- task/tool_class.py (explicit tool class registry)

## Updated
- task/tool.py: adds explicit tool_class
- task/escalation.py: adds MUTATE_WITHIN_CLASS routing + fuses
- task/tool_evolver.py: split within-class mutation vs tool-class escalation
- task/executive.py: wires stall → probe → reframe once → mutate within class → new class → abandon
- task/simulation.py: fixes CapabilityGapDetector.update signature
- main.py + test_tool_evolution.py: normalized imports to core/task/environment packages

## Notes
- No reward signals, no scores, no rankings were added.
- Reframing remains one-shot and Level-1 only for auto-apply.
