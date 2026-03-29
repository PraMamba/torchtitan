---
name: review-pr
description: Intelligent PR code review with dynamic agent allocation based on change types
allowed-tools: Read, Grep, Glob, Bash, Task
---

@.claude/data/review-pr-change-types.md @.claude/data/review-pr-templates.md

# PR Code Review (Dynamic Agent Allocation)

Intelligent code review for the current branch's Pull Request. Dynamically
generates targeted review tasks based on PR changes.

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/review-pr 123`)
- `--quick`: Quick mode, only run Phase 1 analysis

## Model Configuration

| Mode | CRITICAL/HIGH | MEDIUM | LOW |
|------|---------------|--------|-----|
| **Default** | Opus | Sonnet | Haiku |
| **Quick** (`--quick`) | Sonnet | Sonnet | Sonnet |

## Phase 1: Deep PR Analysis

### 1.1 Get PR Summary
```bash
gh pr view --json number,title,body,files,additions,deletions
gh pr diff
```

### 1.2 Change Type Detection
Analyze each file change, detecting types by risk level.
Reference: `review-pr-change-types.md`

### 1.3 Output Change Analysis Report
```
CHANGE_ANALYSIS_REPORT:
- detected_types: [DISTRIBUTED_CORE, TENSOR_PARALLEL, ...]
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- affected_files: [file1.py, ...]
- identified_risks: [risk1, ...]
```

## Phase 2: Dynamic Agent Planning

1. Generate tasks by risk area — each high-risk area gets a dedicated task
2. Merge related changes — interdependent changes can be merged
3. Model selection — CRITICAL/HIGH -> Opus, MEDIUM -> Sonnet, LOW -> Haiku

## Phase 3: Execute Review Tasks (Parallel)

Each agent reviews independently using checklist from `review-pr-templates.md`.

**Agent Output Format:**
```
REVIEW_RESULT:
task_name: "Task Name"
findings:
  - issue: "Issue description"
    severity: CRITICAL | HIGH | MEDIUM | LOW
    file: "path/to/file.py"
    line: 123
    reason: "Why this is an issue"
    suggestion: "Fix suggestion"
```

## Phase 4: Summary Report

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Detected Change Types**: [...]
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW

## Findings

### CRITICAL Severity
...

### HIGH Severity
...

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
```

## Important Notes
- Do NOT check build signals or try to build/type-check
- Use `gh` to interact with GitHub
- Do NOT automatically post comments to PR
- Must provide file path and line number when referencing issues
- Verify numerical impact for any computation changes
