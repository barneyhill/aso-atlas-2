#!/bin/bash
# setup_env.sh - Environment setup for modXNA pipeline
# Source this file to configure your environment:
#   source /path/to/modxna/scripts/setup_env.sh

# Determine script location (works with source command)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODXNA_PIPELINE_HOME="$(dirname "$SCRIPT_DIR")"

# Conda environment
export AMBERHOME="$MODXNA_PIPELINE_HOME/bin/conda_env"

# modXNA installation
export MODXNA_HOME="$MODXNA_PIPELINE_HOME/bin/modXNA"

# Update PATH (cpptraj_install first for newer cpptraj 6.30+)
export PATH="$MODXNA_PIPELINE_HOME/bin/cpptraj_install/bin:$PATH"
export PATH="$MODXNA_PIPELINE_HOME/bin/conda_env/bin:$PATH"
export PATH="$MODXNA_PIPELINE_HOME/bin/modXNA:$PATH"
export PATH="$MODXNA_PIPELINE_HOME/bin:$PATH"

echo "modXNA pipeline environment configured:"
echo "  MODXNA_PIPELINE_HOME: $MODXNA_PIPELINE_HOME"
echo "  AMBERHOME: $AMBERHOME"
echo "  MODXNA_HOME: $MODXNA_HOME"
echo "  PATH updated to include cpptraj_install/bin, conda_env/bin, and modXNA"
