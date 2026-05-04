---
name: create-pr
description: Rebase from the latest origin/main, squash commits, and create a PR on GitHub. Invoke with /create-pr.
---

# Create Pull Request

Rebase from the latest `origin/main`, squash commits, and create a PR on GitHub
with an intelligent title and description.

## Usage

```
/create-pr [--draft] [--base <branch>]
```

## Workflow

### Step 1: Verify Prerequisites

```bash
git branch --show-current
git status --short
gh --version
```

- Must NOT be on main/master branch
- Must have no uncommitted changes (ask user to commit/stash first)

### Step 2: Check for Existing PR

```bash
gh pr view --json number,title,url 2>/dev/null || echo "No existing PR"
```

If PR exists, ask user permission to force-update it.

### Step 3: Fetch and Rebase

```bash
git fetch origin main
git log --oneline HEAD ^origin/main
git rebase origin/main
```

On conflict: `git rebase --abort` and let user handle manually.

### Step 4: Squash Commits

```bash
git rev-list --count origin/main..HEAD
git reset --soft origin/main
```

Generate commit message using `/gen-commit-msg` logic.

### Step 5: Generate PR Title and Description

**PR Title:** `<type>(<scope>): <brief description>` (under 70 chars)

**PR Description Format:**
```markdown
## Summary
[2-4 sentences explaining what and why]

## Changes
| File | Change |
|------|--------|
| path/to/file.py | Description |

## Testing
- [ ] `pre-commit run --all-files` passes
- [ ] `pytest tests/unit_tests/ -x` passes
- [ ] Integration tests verified (if parallelism changes)
- [ ] Numerical validation (if computation changes)

## Numerical Validation
[Include loss comparison if applicable, or "N/A - no computation changes"]
```

### Step 6: Push and Create PR

```bash
git push -f -u origin $(git branch --show-current)

gh pr create \
  --base main \
  --title "<title>" \
  --body "$(cat <<'EOF'
<description>
EOF
)"
```

### Safety Checks

- Confirm not on main/master
- Backup branch before squash
- Show preview of title/description before creating
- Warn about force push rewriting history
