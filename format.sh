!#/bin/bash
PROJ_DIR="$(cd "$(dirname "${0}")" && pwd)"
find . -name "*.py" | xargs yapf -i --style=${PROJ_DIR}/style.cfg
