# HELM to AMBER/modXNA Conversion Pipeline

Automated pipeline to convert HELM-encoded oligonucleotide sequences into AMBER-ready simulation files using modXNA.

## Overview

This tool:
1. Parses HELM notation (e.g., `RNA1{[moe](C)[sp].[moe](T)[sp].d(A)}$$$$`)
2. Maps modifications to modXNA fragment codes
3. Builds parameterized residues using modXNA.sh
4. Generates AMBER topology (.parm7) and coordinate (.crd) files
5. Creates minimization/equilibration input files for MD

## Installation

```bash
# Install AmberTools and modXNA to bin/ (isolated, no system-wide install)
./scripts/install.sh

# Activate environment
source scripts/setup_env.sh
```

Requirements:
- macOS (arm64/x86_64) or Linux
- ~3GB disk space for AmberTools

## Usage

```bash
# Basic usage
uv run python -m modxna.cli "RNA1{[moe](C)[sp].[moe](T)[sp].d(A)}$$$$" -o output/

# From a file
uv run python -m modxna.cli input.helm -o output/

# Options
uv run python -m modxna.cli --help
```

## Scripts

The `scripts/` directory contains utilities for installation and running simulations:

| Script | Description |
|--------|-------------|
| `install.sh` | Install AmberTools and modXNA to `bin/` |
| `setup_env.sh` | Set up environment variables for AMBER |
| `run_example_aso.py` | Run a complete MD simulation with AMBER |

### Running AMBER Simulations

After building a system, you can run MD simulations using AMBER:

```bash
# Run the example ASO simulation (builds system, runs MD, exports video)
uv run python scripts/run_example_aso.py

# Customize simulation settings
uv run python scripts/run_example_aso.py --simulation-ns 10 --threads 8

# Skip steps if you've already run them
uv run python scripts/run_example_aso.py --skip-build --skip-video
```

## HELM Notation Reference

| Notation | Chemistry | modXNA Code |
|----------|-----------|-------------|
| **Sugar** |||
| `d` | 2'-deoxy (DNA) | DC2 |
| `[moe]` | 2'-O-methoxyethyl | MOE |
| `[cet]` | Constrained ethyl (cEt) | CET |
| `[lna]` | Locked nucleic acid | LNA |
| `[fR]` | 2'-fluoro | AF2 |
| `[m]` | 2'-O-methyl | OME |
| (default) | Ribose (RNA) | RC3 |
| **Backbone** |||
| `[sp]` | Phosphorothioate | PS1 |
| `.` | Phosphodiester | RPO |
| `[am]` | Phosphoramidate | NS1 |
| **Base** |||
| `5meC` | 5-methylcytosine | M5C |

## Output Files

```
output/
├── residues/           # Individual residue builds
│   └── XXX/            # Per-residue .lib files
├── combined.lib        # All residues combined
├── aso.parm7           # AMBER topology
├── aso.hmr.parm7       # HMR topology (4fs timestep)
├── aso.crd             # AMBER coordinates
├── aso.pdb             # Structure visualization
└── md_inputs/          # MD simulation files
    ├── min1.in         # Initial minimization
    ├── min2.in         # Full minimization
    ├── heat.in         # Heating (0→300K)
    ├── density.in      # NPT equilibration
    ├── equil1-4.in     # Restraint release
    ├── prod.in         # Production MD
    └── run_md.sh       # Example run script
```

## Simulation Settings

Based on the modXNA paper (Love et al., JCTC 2024):
- **Water model**: OPC
- **Box**: Truncated octahedron, 9.0 Å buffer
- **Ions**: 150 mM NaCl (Joung-Cheatham parameters)
- **Timestep**: 4 fs with HMR
- **Temperature**: 300 K (Langevin thermostat)

## References

- Love O, et al. (2024). modXNA: A Modular Approach to Parametrization of Modified Nucleic Acids. *J. Chem. Theory Comput.* 20:9354-9363.
- ModXNA Tutorial: https://modxna.chpc.utah.edu/
- ModXNA GitHub: https://github.com/drroe/modXNA
