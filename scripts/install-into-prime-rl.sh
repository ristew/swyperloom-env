#!/usr/bin/env bash
# Install the swyperloom env into a prime-rl checkout's venv so
# `uv run rl @ .../configs/rl/swyperloom-1b.toml` resolves the env id.
#
# Usage:
#   scripts/install-into-prime-rl.sh /path/to/prime-rl
set -euo pipefail

PRIME_RL_DIR="${1:-}"
if [[ -z "$PRIME_RL_DIR" ]]; then
  echo "usage: $0 /path/to/prime-rl" >&2
  exit 1
fi
if [[ ! -d "$PRIME_RL_DIR" ]]; then
  echo "error: $PRIME_RL_DIR is not a directory" >&2
  exit 1
fi

SWYPERLOOM_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Installing swyperloom env ($SWYPERLOOM_DIR) into prime-rl venv ($PRIME_RL_DIR) ..."
(cd "$PRIME_RL_DIR" && uv pip install -e "$SWYPERLOOM_DIR")

echo
echo "Verifying import..."
(cd "$PRIME_RL_DIR" && uv run python -c "import swyperloom; print('  swyperloom OK')")

echo
echo "Done. To run 1B local training:"
echo "  cd $PRIME_RL_DIR"
echo "  bash scripts/tmux.sh"
echo "  # Launcher pane:"
echo "  uv run inference --model.name meta-llama/Llama-3.2-1B &"
echo "  uv run rl @ $SWYPERLOOM_DIR/configs/rl/swyperloom-1b.toml"
