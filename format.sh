#!/bin/bash
find core -name "*.py" | xargs isort
find core -name "*.py" | xargs yapf -i --style=style.cfg
