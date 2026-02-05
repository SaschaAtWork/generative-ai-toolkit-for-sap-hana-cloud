#!/usr/bin/env bash
# Cloud Foundry pre-runtime hook to install local package

echo "-----> Installing local hana_ai package..."
python -m pip install -e .
echo "-----> Local package installation complete"
