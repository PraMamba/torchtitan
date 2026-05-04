# Hook Non-Migration Reference

The original Claude workflow contains hook assets, but this migration does **not**
translate them into executable Codex hooks.

Reason: the approved requirements explicitly made hook migration a non-goal. This
file preserves the source hook content as reference-only documentation so the
behavior is not silently lost.

Do not add executable lifecycle-hook configuration under `.codex` as part of this compatibility layer.

## Source: `.claude/hooks/check-expert-update.sh`

```bash
#!/bin/bash
# Hook: Remind Claude to update expert agents when related code changes
# Triggered by PostToolUse on Write/Edit operations
# Reads JSON input from stdin (Claude Code hook interface)

# Check if jq is available
if ! command -v jq &> /dev/null; then
    exit 0
fi

# Read JSON input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Map changed files to relevant expert agents
check_expert_update() {
    local file="$1"
    local reminder_file=""
    local reminder_desc=""

    # Distributed training code
    if [[ "$file" == *"torchtitan/distributed/"* ]]; then
        reminder_file="distributed-expert.md"
        reminder_desc="Distributed/Parallelism"
    fi

    # Model code (common and specific)
    if [[ "$file" == *"torchtitan/models/common/"* ]] || \
       [[ "$file" == *"torchtitan/models/llama3/"* ]] || \
       [[ "$file" == *"torchtitan/models/llama4/"* ]] || \
       [[ "$file" == *"torchtitan/models/qwen3/"* ]] || \
       [[ "$file" == *"torchtitan/models/deepseek_v3/"* ]] || \
       [[ "$file" == *"torchtitan/models/gpt_oss/"* ]] || \
       [[ "$file" == *"torchtitan/models/flux/"* ]]; then
        reminder_file="model-expert.md"
        reminder_desc="Model Architecture"
    fi

    # Config and protocol code
    if [[ "$file" == *"torchtitan/config/"* ]] || \
       [[ "$file" == *"torchtitan/protocols/"* ]]; then
        reminder_file="config-expert.md"
        reminder_desc="Config/Protocol"
    fi

    # Checkpoint code
    if [[ "$file" == *"torchtitan/components/checkpoint.py"* ]] || \
       [[ "$file" == *"torchtitan/protocols/state_dict_adapter.py"* ]] || \
       [[ "$file" == *"torchtitan/protocols/model_converter.py"* ]]; then
        reminder_file="checkpoint-expert.md"
        reminder_desc="Checkpoint/StateDictAdapter"
    fi

    # Trainer and component code
    if [[ "$file" == *"torchtitan/trainer.py"* ]] || \
       [[ "$file" == *"torchtitan/train.py"* ]] || \
       [[ "$file" == *"torchtitan/components/"* ]]; then
        reminder_file="trainer-expert.md"
        reminder_desc="Trainer/Components"
    fi

    # Output reminder if matched
    if [ -n "$reminder_file" ]; then
        echo ""
        echo "REMINDER: Modified $reminder_desc code ($file)."
        echo "Consider updating: .claude/agents/$reminder_file"
        echo ""
    fi
}

check_expert_update "$FILE_PATH"
```
