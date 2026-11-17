#!/usr/bin/env bash
# Observe one Kaggle Cycle 5 kernel every 10 minutes until completion/failure.
# This runs on the agent's authenticated machine; it does not train locally.
set -euo pipefail

kernel="${1:-jayanandj/moonshot-cycle5-gpu}"
observations="${2:-cycle5-kaggle-monitor}"
poll_seconds="${MOONSHOT_POLL_SECONDS:-600}"
max_polls="${MOONSHOT_MAX_POLLS:-72}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "[ERROR] Kaggle CLI is missing. Install/login through the normal Kaggle workflow." >&2
  exit 2
fi
if ! [[ "$poll_seconds" =~ ^[0-9]+$ && "$max_polls" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Poll settings must be nonnegative integers." >&2
  exit 2
fi
if (( poll_seconds < 1 || max_polls < 1 )); then
  echo "[ERROR] Poll interval and count must be positive." >&2
  exit 2
fi

mkdir -p "$observations"
printf 'checked_utc\tstatus\n' > "$observations/status_history.tsv"

for ((attempt=1; attempt<=max_polls; attempt++)); do
  now="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if status="$(kaggle kernels status "$kernel" 2>&1)"; then
    state="$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]')"
    printf '%s\t%s\n' "$now" "${status//$'\n'/ }" >> "$observations/status_history.tsv"
    echo "[$now] Check $attempt/$max_polls: $status"

    # Save the latest complete log snapshot; never append duplicate log copies.
    if kaggle kernels logs "$kernel" > "$observations/latest_kaggle.log" 2>&1; then
      echo "[$now] Refreshed $observations/latest_kaggle.log"
    else
      echo "[$now] Logs temporarily unavailable; continuing."
    fi

    if [[ "$state" =~ (^|[^a-z])(complete|completed|success)([^a-z]|$) ]]; then
      echo "[$now] Kaggle reports completion. Downloading evidence."
      mkdir -p "$observations/downloaded"
      kaggle kernels output "$kernel" -p "$observations/downloaded" --force
      if ! find "$observations/downloaded" -type f -name 'run_manifest.json' -print -quit | grep -q .; then
        echo "[ERROR] Kaggle completed but no Cycle 5 run_manifest.json was downloaded." >&2
        exit 3
      fi
      if ! find "$observations/downloaded" -type f -name 'smoke_result.json' -print -quit | grep -q .; then
        echo "[ERROR] Kaggle completed but GPU smoke evidence is missing." >&2
        exit 3
      fi
      echo "[$now] Run finished; verify run_manifest.json status and scientific gate before reporting success."
      exit 0
    fi

    if [[ "$state" =~ (^|[^a-z])(error|errored|fail|failed|cancelled|canceled)([^a-z]|$) ]]; then
      echo "[ERROR] Kaggle reported a terminal failure: $status" >&2
      mkdir -p "$observations/downloaded"
      kaggle kernels output "$kernel" -p "$observations/downloaded" --force || true
      exit 1
    fi
  else
    printf '%s\t%s\n' "$now" "STATUS_QUERY_ERROR" >> "$observations/status_history.tsv"
    echo "[$now] Could not query status; will retry (transient error or missing auth)."
  fi

  if (( attempt < max_polls )); then
    echo "[$now] Still running or queued. Checking again in ${poll_seconds}s."
    sleep "$poll_seconds"
  fi
done

echo "[ERROR] Stopped monitoring after $max_polls checks without a terminal status; run outcome is UNKNOWN." >&2
exit 124
