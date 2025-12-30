"""Generate MD input files following AmberMDPrep protocol."""

from pathlib import Path


def generate_md_inputs(
    output_dir: Path,
    topology_name: str = "aso",
    simulation_ns: float = 100.0,
    n_threads: int = 1,
):
    """Generate minimization/equilibration/production input files.

    Args:
        output_dir: Base output directory
        topology_name: Base name for topology files (default: "aso")
        simulation_ns: Production simulation length in nanoseconds (default: 100)
        n_threads: Number of CPU threads for OpenMP (default: 1)

    Creates md_inputs/ directory with:
        - min1.in: Restrained minimization
        - min2.in: Full minimization
        - heat.in: Heating NVT (0 -> 300K)
        - density.in: NPT equilibration
        - equil1-4.in: Gradual restraint release
        - prod.in: Production NVT with 4fs timestep
        - run_md.sh: Example run script
    """
    md_dir = output_dir / "md_inputs"
    md_dir.mkdir(exist_ok=True)

    # Write all input files
    _write_min1(md_dir)
    _write_min2(md_dir)
    _write_heat(md_dir)
    _write_density(md_dir)
    _write_equil_stages(md_dir)
    _write_prod(md_dir, simulation_ns=simulation_ns)
    _write_run_script(md_dir, output_dir, topology_name, n_threads=n_threads)

    print(f"MD input files written to {md_dir}")
    print(f"  Production simulation: {simulation_ns} ns ({int(simulation_ns * 250000)} steps)")


def _write_min1(md_dir: Path):
    """Write restrained minimization input file."""
    content = """\
Minimization 1: Restrained minimization (solvent only)
 &cntrl
    imin=1,
    ntx=1,
    irest=0,
    maxcyc=5000,
    ncyc=2500,
    ntpr=100,
    ntwx=0,
    ntwr=500,
    cut=9.0,
    ntr=1,
    restraintmask='!(:WAT,Na+,Cl-)',
    restraint_wt=10.0,
 /
"""
    with open(md_dir / "min1.in", "w") as f:
        f.write(content)


def _write_min2(md_dir: Path):
    """Write full minimization input file."""
    content = """\
Minimization 2: Full system minimization
 &cntrl
    imin=1,
    ntx=1,
    irest=0,
    maxcyc=10000,
    ncyc=5000,
    ntpr=100,
    ntwx=0,
    ntwr=1000,
    cut=9.0,
    ntr=0,
 /
"""
    with open(md_dir / "min2.in", "w") as f:
        f.write(content)


def _write_heat(md_dir: Path):
    """Write heating (NVT) input file."""
    content = """\
Heating: NVT 0->300K over 100ps
 &cntrl
    imin=0,
    ntx=1,
    irest=0,
    nstlim=50000,
    dt=0.002,
    ntf=2,
    ntc=2,
    tempi=0.0,
    temp0=300.0,
    ntpr=500,
    ntwx=500,
    ntwr=5000,
    cut=9.0,
    ntr=1,
    restraintmask='!(:WAT,Na+,Cl-)',
    restraint_wt=10.0,
    ntt=3,
    gamma_ln=2.0,
    ig=-1,
    ntb=1,
    nmropt=1,
 /
 &wt type='TEMP0', istep1=0, istep2=50000, value1=0.0, value2=300.0 /
 &wt type='END' /
"""
    with open(md_dir / "heat.in", "w") as f:
        f.write(content)


def _write_density(md_dir: Path):
    """Write NPT equilibration input file."""
    content = """\
Density equilibration: NPT at 300K for 500ps
 &cntrl
    imin=0,
    ntx=5,
    irest=1,
    nstlim=250000,
    dt=0.002,
    ntf=2,
    ntc=2,
    temp0=300.0,
    ntpr=500,
    ntwx=500,
    ntwr=10000,
    cut=9.0,
    ntr=1,
    restraintmask='!(:WAT,Na+,Cl-)',
    restraint_wt=5.0,
    ntt=3,
    gamma_ln=2.0,
    ig=-1,
    ntb=2,
    ntp=1,
    barostat=2,
    pres0=1.0,
    taup=2.0,
 /
"""
    with open(md_dir / "density.in", "w") as f:
        f.write(content)


def _write_equil_stages(md_dir: Path):
    """Write equilibration stages with decreasing restraints."""
    restraint_weights = [2.0, 1.0, 0.5, 0.1]

    for i, wt in enumerate(restraint_weights, 1):
        content = f"""\
Equilibration stage {i}: NPT with {wt} kcal/mol/A^2 restraints, 500ps
 &cntrl
    imin=0,
    ntx=5,
    irest=1,
    nstlim=250000,
    dt=0.002,
    ntf=2,
    ntc=2,
    temp0=300.0,
    ntpr=500,
    ntwx=500,
    ntwr=10000,
    cut=9.0,
    ntr=1,
    restraintmask='!(:WAT,Na+,Cl-)',
    restraint_wt={wt},
    ntt=3,
    gamma_ln=2.0,
    ig=-1,
    ntb=2,
    ntp=1,
    barostat=2,
    pres0=1.0,
    taup=2.0,
 /
"""
        with open(md_dir / f"equil{i}.in", "w") as f:
            f.write(content)


def _write_prod(md_dir: Path, simulation_ns: float = 100.0):
    """Write production NVT input file with 4fs timestep (requires HMR).

    Args:
        md_dir: Directory for MD input files
        simulation_ns: Simulation length in nanoseconds
    """
    # Calculate nstlim: ns * 1e6 ps/ns / 0.004 ps/step = ns * 250000 steps
    nstlim = int(simulation_ns * 250000)
    # Output frequency: save frame every 10ps (2500 steps at 4fs timestep)
    ntwx = 2500
    content = f"""\
Production: NVT at 300K with 4fs timestep (HMR topology required)
Duration: {simulation_ns} ns ({nstlim} steps)
 &cntrl
    imin=0,
    ntx=5,
    irest=1,
    nstlim={nstlim},
    dt=0.004,
    ntf=2,
    ntc=2,
    temp0=300.0,
    ntpr=5000,
    ntwx={ntwx},
    ntwr=50000,
    cut=9.0,
    ntr=0,
    ntt=3,
    gamma_ln=2.0,
    ig=-1,
    ntb=1,
    iwrap=1,
 /
"""
    with open(md_dir / "prod.in", "w") as f:
        f.write(content)


def _write_run_script(
    md_dir: Path, output_dir: Path, topology_name: str, n_threads: int = 1
):
    """Write example run script for MD simulation.

    Args:
        md_dir: Directory for MD input files
        output_dir: Base output directory
        topology_name: Base name for topology files
        n_threads: Number of CPU threads for OpenMP
    """
    content = f"""\
#!/bin/bash
# MD run script for modXNA oligonucleotide
# Generated by modXNA pipeline
#
# Usage: ./run_md.sh
# Requires: sander from AmberTools (or pmemd.cuda for GPU)
#
# This script runs:
# 1. Minimization (2 stages)
# 2. Heating (NVT, 0->300K)
# 3. Density equilibration (NPT)
# 4. Equilibration with restraint release (4 stages)
# 5. Production (NVT with 4fs timestep)

set -e

# Configuration
export OMP_NUM_THREADS={n_threads}
SANDER="sander"  # CPU execution (use "pmemd.cuda" for GPU)
PRMTOP="{output_dir}/{topology_name}.parm7"
PRMTOP_HMR="{output_dir}/{topology_name}.hmr.parm7"
CRD="{output_dir}/{topology_name}.crd"

# Create trajectory directory
mkdir -p trajectories

echo "=== Running MD with ${{OMP_NUM_THREADS}} CPU threads ==="
echo ""

echo "=== Minimization 1: Restrained ==="
$SANDER -O -i min1.in -o min1.out -p $PRMTOP -c $CRD -r min1.rst7 -ref $CRD

echo "=== Minimization 2: Full system ==="
$SANDER -O -i min2.in -o min2.out -p $PRMTOP -c min1.rst7 -r min2.rst7

echo "=== Heating: 0->300K ==="
$SANDER -O -i heat.in -o heat.out -p $PRMTOP -c min2.rst7 -r heat.rst7 -x trajectories/heat.nc -ref min2.rst7

echo "=== Density equilibration ==="
$SANDER -O -i density.in -o density.out -p $PRMTOP -c heat.rst7 -r density.rst7 -x trajectories/density.nc -ref heat.rst7

echo "=== Equilibration stages ==="
PREV_RST="density.rst7"
for i in 1 2 3 4; do
    echo "  Stage $i..."
    $SANDER -O -i equil$i.in -o equil$i.out -p $PRMTOP -c $PREV_RST -r equil$i.rst7 -x trajectories/equil$i.nc -ref $PREV_RST
    PREV_RST="equil$i.rst7"
done

echo "=== Production (using HMR topology for 4fs timestep) ==="
$SANDER -O -i prod.in -o prod.out -p $PRMTOP_HMR -c equil4.rst7 -r prod.rst7 -x trajectories/prod.nc

echo "=== Done! ==="
echo "Production trajectory: trajectories/prod.nc"
"""
    script_path = md_dir / "run_md.sh"
    with open(script_path, "w") as f:
        f.write(content)

    # Make executable
    script_path.chmod(0o755)
