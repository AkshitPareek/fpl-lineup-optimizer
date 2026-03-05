# Execution Plan (Architect-Owned)

## Objective
Deliver ML Phase 2 through short, isolated tasks with strict test-gated completion.

## Global Rule
A task can be marked complete only when all mapped tests pass.

## Sequence and Dependencies
1. P2-T1 Baseline Models (A)
2. P2-T2 Advanced Features (A) depends on P2-T1
3. P2-T3 Gradient Boosting (B) depends on P2-T2
4. P2-T4 Neural Network (C, optional) depends on P2-T2
5. P2-T5 Ensemble (B) depends on P2-T3 and optionally P2-T4
6. P2-T6 Evaluation (C) depends on P2-T5

## Required Workflow Per Task
1. Architect writes task brief.
2. Implementing teammate writes short plan.
3. Architect signs off plan.
4. In parallel, another teammate prepares:
   - subtask checklist
   - test harness updates
5. Implementing teammate codes in isolated worktree and branch.
6. Teammate runs gate command:
   - `./backend/scripts/verify_task_completion.sh <TASK_ID>`
7. Open PR to `dev/ml-phase2` with test output.
8. Reviewer signs off.
9. Lead merges.

## Worktree Model
- Base branch: `dev/ml-phase2`
- One worktree per active teammate
- One feature branch per task

## Suggested Parallelization
- While A executes P2-T1, Test Harness Teammate prepares P2-T2 and P2-T3 harnesses.
- While B/C execute P2-T3/P2-T4, Test Harness Teammate prepares P2-T5/P2-T6 harnesses.

## Completion Evidence Required In PR
- Task ID and scope summary
- Architect sign-off link or note
- Test harness changes (if any)
- Output of mapped tests
- Known risks and rollback note
