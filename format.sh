#!/bin/bash
find core -name "*.py" | xargs autopep8 --in-place --aggressive
find core -name "*.py" | xargs yapf -i --style=style.cfg
