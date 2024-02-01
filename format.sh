#!/bin/bash
PROJ_DIR="$(cd "$(dirname "${0}")" && pwd)"
find core -name "*.py" | xargs autopep8 --in-place --aggressive --aggressive
find core -name "*.py" | xargs yapf -i --style=${PROJ_DIR}/style.cfg
