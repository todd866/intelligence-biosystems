# Turn-Taking System

This directory uses flag files to coordinate turn-taking between Codex and Claude Code.

## How It Works

**Only ONE of these files should exist at a time:**

- `TURN_codex.txt` - Codex should run next (analyze and review)
- `TURN_claude.txt` - Claude Code should run next (implement fixes)

## Workflow

### When Codex runs:
1. Checks for `TURN_codex.txt` (if missing, don't run)
2. Performs read-only analysis
3. Writes findings to `codex.txt`
4. Deletes `TURN_codex.txt`
5. Creates `TURN_claude.txt`

### When Claude Code runs:
1. Checks for `TURN_claude.txt` (if missing, don't run)
2. Reads `codex.txt` commentary
3. Implements fixes
4. Appends changelog to `codex.txt`
5. Deletes `TURN_claude.txt`
6. Creates `TURN_codex.txt`

## Manual Override

You (Ian) can manually switch turns by:

```bash
# Give turn to Codex:
cd ~/Desktop/highdimensional/3_intelligence/commentary
rm -f TURN_claude.txt
touch TURN_codex.txt

# Give turn to Claude Code:
rm -f TURN_codex.txt
touch TURN_claude.txt
```

## Current Status

Check whose turn it is:
```bash
ls -1 ~/Desktop/highdimensional/3_intelligence/commentary/TURN_*.txt
```

## Integration with Git

After each turn, commit changes:
```bash
git add .
git commit -m "Codex: review findings - 3 issues"
# or
git commit -m "Claude: fixed issues 1-2, compiled successfully"
```

This creates an audit trail of all changes.
