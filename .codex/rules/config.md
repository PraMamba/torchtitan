# Codex Rule Reference: config

Migrated from `.claude/rules/config.md`.

This file preserves Claude rule content as a Codex reference. The original
`globs` metadata describes the intended source-code scope; because this file
lives under `.codex/`, it does not by itself enforce rules outside `.codex/`.

## Original Claude rule metadata

```yaml
description: Rules for the config system
globs: torchtitan/**
```

---


# Config System Rules

## No None Defaults for Required Fields
If a config field is required, don't give it a `None` default. Either set a
meaningful default or make it a required argument with no default.

## Parameter Ordering
- Put important, commonly-used fields first.
- Less important or rarely-changed fields go later.
- Prefer keyword-only arguments after the first positional arg to prevent
  positional mistakes.

## Naming Conventions
- Use descriptive names that reflect what the config controls.
- Use `num_` prefix for count fields.
- Prefer strings as config names when they benefit extensibility (e.g. model
  selection, dataset selection).

## Safety
- When config values are passed through to external APIs (e.g. DataLoader kwargs),
  ensure all values are safe for all code paths (e.g. what happens when
  `num_workers == 0`?).
- When a config option silently doesn't take effect in certain code paths,
  emit a warning to the user.
