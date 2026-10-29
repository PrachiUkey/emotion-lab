#!/bin/bash
# Builds only the NED env. The full setup (both envs, checkpoints, dashboard) is ../setup.sh.
exec "$(dirname "$0")/../setup.sh" --skip-lab --skip-ckpt --skip-web "$@"
