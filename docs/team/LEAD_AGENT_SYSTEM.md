# Lead Agent Operating System

This document is the source of truth for the multi-agent workflow in this repo.

## Scope
- Project branch: `dev/ml-phase2`
- Team mode: lead-managed, task-oriented, test-gated delivery
- Completion rule: no task is complete unless all mapped tests pass

## Hard Rules
1. Lead creates and protects `dev/ml-phase2` as integration branch.
2. Architect produces the execution plan before teammate implementation starts.
3. Every teammate creates a short implementation plan and gets architect sign-off before coding.
4. In parallel to architect sign-off, another teammate writes or refines the subtask checklist and test harness for that task.
5. Teammates work only in personal worktrees and feature branches.
6. A task can move to `Done` only if:
   - teammate-local mapped tests pass
   - reviewer confirms changes and risks
   - merge target remains green after merge
7. Reviewer is mandatory before merge.
8. Devil's Advocate is mandatory at the end and must challenge complexity, unnecessary scope, and hidden risk.

## Role Definitions
- Lead: orchestration, branch strategy, docs/rules/memory upkeep, merge authority.
- Architect: optimized plan, dependencies, and acceptance criteria.
- Teammates (A/B/C...): implementation in isolated worktrees.
- Test Harness Teammate: creates and maintains task test harnesses; blocks completion when failing.
- Reviewer: code and regression review before merge.
- Devil's Advocate: final simplification and risk challenge.

## Standard Lifecycle Per Task
1. Architect issues task brief.
2. Implementing teammate submits plan (short, task-scoped).
3. Parallel teammate drafts subtask checklist + tests for same task.
4. Architect approves plan.
5. Implementing teammate writes code in worktree.
6. Run mapped tests via `backend/scripts/verify_task_completion.sh <TASK_ID>`.
7. Open PR to `dev/ml-phase2` with test evidence.
8. Reviewer approves or requests changes.
9. Lead merges.

## Worktree and Branch Conventions
- Worktree path: `../fpl-ml-<role-or-name>`
- Feature branch pattern: `feature/<TASK_ID>-<short-description>`
- PR target: `dev/ml-phase2`

## Command Snippets
```bash
# Create teammate worktree
git worktree add ../fpl-ml-teammate-a dev/ml-phase2

# In worktree, create feature branch
git checkout -b feature/P2-T1-baseline-models

# Verify task completion gate
./backend/scripts/verify_task_completion.sh P2-T1
```

## Memory Maintenance (Lead)
Keep these files current:
- `docs/team/LEAD_CONTEXT.md`
- `backend/TASK_TRACKING.md`
- `docs/team/TASK_TEST_MAP.md`
