# Devil's Advocate Report (Teammate: Devil's Advocate)

Date: 2026-03-05
Scope: Final pre-merge challenge for P2-T1/2/3/5/6

## Challenges
1. Are we over-coupling ensemble behavior to local dataset files?
- Risk: `EnsemblePredictor` auto-calibration from `datasets/fpl_points_v1/validation_*` can create hidden behavior shifts.
- Simplify: make calibration explicit via a method call or constructor flag.

2. Are evaluation utilities too permissive?
- Risk: tests pass, but report methods are generic and may not reflect true FPL domain constraints.
- Simplify: lock report schema and metric set in one config object.

3. Is optional task handling explicit enough?
- Risk: P2-T4 may be forgotten while remaining blocked.
- Simplify: add a formal "optional deferred" status with owner/date rationale.

## Final Recommendation
- Merge P2-T1/2/3/5/6 to `dev/ml-phase2` after lead confirms no extra scope.
- Defer P2-T4 with explicit dependency note (`torch`) and re-entry criteria.
