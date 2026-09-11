#!/usr/bin/env bash
# Enforces the persisted seven-hour deadline independently of the conversation.
# Writes a STOP sentinel the runners poll, then kills any remaining local worker.
set -u
STATE=out/self_focus/SESSION.json
STOP=out/self_focus/STOP
DEADLINE=$(python -c "import json;print(int(json.load(open(r'$STATE'))['deadline_epoch']))")
while :; do
  NOW=$(python -c "import time;print(int(time.time()))")
  if [ "$NOW" -ge "$DEADLINE" ]; then
    date +"%Y-%m-%d %H:%M:%S deadline reached; writing STOP" >> out/self_focus/watchdog.log
    echo "deadline $DEADLINE reached at $NOW" > "$STOP"
    # kill only this session's workers, by module name
    for pid in $(wmic process where "name like '%python%'" get processid,commandline 2>/dev/null \
                 | grep -i "experiments.self_focus" | grep -oE "[0-9]+$"); do
      echo "killing $pid" >> out/self_focus/watchdog.log
      taskkill //PID "$pid" //F >> out/self_focus/watchdog.log 2>&1
    done
    exit 0
  fi
  sleep 60
done
