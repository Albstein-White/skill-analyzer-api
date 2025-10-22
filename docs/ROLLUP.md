# SkillTier Adaptive Rollout Checklist

## Feature Flags
- `PLAN_ENABLED`: controls post-run improvement plan generation. Set `PLAN_ENABLED=0` to disable in staging.
- `OPEN_ENABLED_LONG`: toggles OPEN scheduling for long adaptive runs. Use `OPEN_ENABLED_LONG=0` to keep long runs objective-only.
- Flags honour environment overrides via `_env_bool` helpers; defaults stay conservative for production.

## API Notes
- `/session/{sid}/finish` now returns `stop_reason`, usage counts, and (in staging) objective fail-fast window lengths for downstream telemetry.
- When `PROD_GOD_ENABLE=1`, long runs may schedule a GOD probe OPEN once SS gates and rubric thresholds are met. Probe metadata and effective step counts surface via finish payloads and `/dev/run` for staging validation.
- GOD probe telemetry includes `god_probe_enabled`, per-domain `first_rubric`, `first_difficulty`, `probe2_rubric`, `probe2_served`, and a `probe_reason` flag (`blocked_progress`, `blocked_fail_fast`, `probe2_low_rubric`, `god_awarded`). `effective_steps` subtracts probe-2 items when `GOD_PROBE_EXEMPT_FROM_CAP` is true.

## Caps and Objective Minima
- Run caps default to `CAP_SHORT=48` and `CAP_LONG=160` steps. Override with environment variables if staging requires shorter passes.
- Objective minima remain fixed at `SHORT_OBJ_MIN=3` and `OBJ_MIN_LONG=8` per domain. The engine always fulfils these before SR/OPEN and marks runs incomplete if caps are reached first.

## Operational Commands
- Bank coverage audit: `python -m skill_core.audit_bank` (writes `/tmp/bank_audit.json` and exits non-zero on warnings).
- Smoke validation: `python -m skill_core.smoke` (set `STAGING_PROFILE=1` for a concise per-domain summary line plus cap status).
- Plan endpoint smoke: `uvicorn api.app:app --reload` then `curl -X POST http://localhost:8000/results/<id>/plan` (use `PLAN_ENABLED=0` to verify disablement).

## Staging Environment Example
```
export PLAN_ENABLED=1
export OPEN_ENABLED_LONG=1
export CAP_LONG=140
export STAGING_PROFILE=1
python -m skill_core.audit_bank
python -m skill_core.smoke
```
