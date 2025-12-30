"""Main CLI entry point for modXNA pipeline."""

import argparse
import sys
from pathlib import Path

from .helm_parser import parse_helm
from .modxna_mapper import map_to_modxna, load_catalog, get_unique_residues
from .residue_builder import build_all_residues
from .amber_builder import (
    generate_combine_lib_script,
    generate_build_script,
    generate_hmr_script,
    run_tleap,
    run_parmed,
    run_minimization,
    set_amber_env,
)
from .md_prep import generate_md_inputs


def main():
    """Main entry point for the modXNA CLI."""
    parser = argparse.ArgumentParser(
        description="Convert HELM notation to AMBER simulation files using modXNA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  modxna "RNA1{[moe](C)[sp].[moe](T)[sp].d(A)}$$$$" -o output/
  modxna input.helm -o output/ --no-solvate
  modxna sequence.helm -o output/ --salt 0.2

Output files:
  output/
  ├── residues/           # Individual residue builds
  ├── combined.lib        # Combined residue library
  ├── aso.parm7           # AMBER topology
  ├── aso.hmr.parm7       # HMR topology (4fs timestep)
  ├── aso.crd             # AMBER coordinates
  ├── aso.pdb             # Structure visualization
  └── md_inputs/          # MD simulation input files
""",
    )

    parser.add_argument(
        "helm",
        help="HELM string or path to .helm file",
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--no-solvate",
        action="store_true",
        help="Skip solvation (generate gas-phase structure only)",
    )
    parser.add_argument(
        "--salt",
        type=float,
        default=0.15,
        help="Salt concentration in M (default: 0.15 for 150mM NaCl)",
    )
    parser.add_argument(
        "--modxna-path",
        type=Path,
        default=None,
        help="Path to modXNA installation (default: auto-detect from MODXNA_HOME or PATH)",
    )
    parser.add_argument(
        "--frcmod",
        type=Path,
        default=None,
        help="Path to frcmod.modxna (default: use bundled or from modXNA installation)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to modxna_catalog.json (default: use bundled catalog)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse HELM and show mapping without building",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-minimize",
        action="store_true",
        help="Skip minimization step (produce only linearized structure)",
    )
    parser.add_argument(
        "--simulation-ns",
        type=float,
        default=100.0,
        help="Production simulation length in nanoseconds (default: 100)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads for MD simulation (default: 1)",
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine HELM input
    helm_input = args.helm
    if Path(helm_input).exists() and Path(helm_input).suffix == ".helm":
        helm_string = Path(helm_input).read_text().strip()
        if args.verbose:
            print(f"Read HELM from file: {helm_input}")
    else:
        helm_string = helm_input

    print(f"Input HELM: {helm_string}")
    print(f"Output directory: {output_dir}")

    # Step 1: Parse HELM
    print("\n=== Step 1: Parsing HELM notation ===")
    try:
        monomers = parse_helm(helm_string)
    except ValueError as e:
        print(f"Error parsing HELM: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(monomers)} monomers")
    if args.verbose:
        for i, m in enumerate(monomers):
            print(f"  {i+1}: {m}")

    # Step 2: Load catalog and map to modXNA
    print("\n=== Step 2: Mapping to modXNA residues ===")
    catalog = load_catalog(args.catalog)
    residues = map_to_modxna(monomers, catalog)
    unique_residues = get_unique_residues(residues)

    print(f"Mapped to {len(residues)} residues ({len(unique_residues)} unique)")
    if args.verbose:
        for resname, r in unique_residues.items():
            print(f"  {resname}: {r.backbone} {r.sugar} {r.base}")

    # Get sequence of residue names for tleap
    sequence = [r.resname for r in residues]
    print(f"Sequence: {' '.join(sequence)}")

    if args.dry_run:
        print("\n=== Dry run complete ===")
        sys.exit(0)

    # Find modXNA path
    modxna_path = _find_modxna_path(args.modxna_path)
    if modxna_path is None:
        print("Error: Could not find modXNA installation.", file=sys.stderr)
        print("Set MODXNA_HOME environment variable or use --modxna-path", file=sys.stderr)
        sys.exit(1)
    print(f"Using modXNA: {modxna_path}")

    # Set up environment for AmberTools
    set_amber_env(modxna_path)

    # Find frcmod path
    frcmod_path = _find_frcmod_path(args.frcmod, modxna_path)
    if frcmod_path is None:
        print("Error: Could not find frcmod.modxna", file=sys.stderr)
        sys.exit(1)
    print(f"Using frcmod: {frcmod_path}")

    # Step 3: Build residues
    print("\n=== Step 3: Building residue libraries ===")
    lib_paths = build_all_residues(list(unique_residues.values()), output_dir, modxna_path)
    print(f"Built {len(lib_paths)} residue libraries")

    # Step 4: Generate and run tleap scripts
    print("\n=== Step 4: Generating AMBER files ===")

    # Combine libraries
    combine_script = generate_combine_lib_script(lib_paths, output_dir)
    print(f"Running: {combine_script}")
    if not run_tleap(combine_script):
        print("Error: Failed to combine libraries", file=sys.stderr)
        sys.exit(1)

    # Build oligonucleotide
    build_script = generate_build_script(
        sequence=sequence,
        output_dir=output_dir,
        frcmod_path=frcmod_path,
        solvate=not args.no_solvate,
        salt_conc=args.salt,
    )
    print(f"Running: {build_script}")
    if not run_tleap(build_script):
        print("Error: Failed to build oligonucleotide", file=sys.stderr)
        sys.exit(1)

    # Apply HMR for 4fs timestep
    print("\n=== Applying hydrogen mass repartitioning ===")
    hmr_script = generate_hmr_script(output_dir)
    print(f"Running: {hmr_script}")
    if not run_parmed(hmr_script):
        print("Warning: Failed to apply HMR (production MD will need 2fs timestep)")

    # Step 5: Generate MD input files
    print("\n=== Step 5: Generating MD input files ===")
    generate_md_inputs(
        output_dir,
        simulation_ns=args.simulation_ns,
        n_threads=args.threads,
    )

    # Step 6: Run minimization (unless skipped)
    if not args.no_minimize:
        print("\n=== Step 6: Running minimization ===")
        if not run_minimization(output_dir):
            print("Warning: Minimization failed. aso.pdb may have clashes.")
            print("You can run minimization manually using md_inputs/min1.in and min2.in")
    else:
        print("\n=== Minimization skipped (--no-minimize) ===")

    # Summary
    print("\n=== Complete! ===")
    print(f"Output files in: {output_dir}")
    print(f"  Topology:     {output_dir}/aso.parm7")
    print(f"  HMR Topology: {output_dir}/aso.hmr.parm7")
    print(f"  Coordinates:  {output_dir}/aso.crd")
    if not args.no_minimize and (output_dir / "aso.min.pdb").exists():
        print(f"  Structure:    {output_dir}/aso.min.pdb (minimized, clash-free)")
    else:
        print(f"  Structure:    {output_dir}/aso.pdb (linearized, may have clashes)")
    print(f"  MD inputs:    {output_dir}/md_inputs/")


def _find_modxna_path(user_path: Path | None) -> Path | None:
    """Find the modXNA installation path."""
    import os
    import shutil

    # User-specified path takes precedence
    if user_path is not None:
        if user_path.exists():
            return user_path
        return None

    # Check MODXNA_HOME environment variable
    modxna_home = os.environ.get("MODXNA_HOME")
    if modxna_home:
        path = Path(modxna_home)
        if path.exists():
            return path

    # Check in project's bin/ directory (from src/modxna/ -> modxna/bin/modXNA)
    project_bin = Path(__file__).parent.parent.parent / "bin" / "modXNA"
    if (project_bin / "modxna.sh").exists():
        return project_bin

    # Check if modxna.sh is in PATH
    modxna_sh = shutil.which("modxna.sh")
    if modxna_sh:
        return Path(modxna_sh).parent

    # Check common locations
    common_paths = [
        Path.home() / "modXNA",
        Path.home() / ".local" / "modXNA",
        Path("/opt/modXNA"),
        Path("/usr/local/modXNA"),
    ]
    for path in common_paths:
        if (path / "modxna.sh").exists():
            return path

    return None


def _find_frcmod_path(user_path: Path | None, modxna_path: Path) -> Path | None:
    """Find the frcmod.modxna parameter file."""
    # User-specified path takes precedence
    if user_path is not None:
        if user_path.exists():
            return user_path
        return None

    # Check in modXNA installation
    candidates = [
        modxna_path / "dat" / "frcmod.modxna",  # Standard location
        modxna_path / "frcmod.modxna",
        modxna_path / "data" / "frcmod.modxna",
        modxna_path / "params" / "frcmod.modxna",
    ]
    for path in candidates:
        if path.exists():
            return path

    # Check bundled in this package
    package_dir = Path(__file__).parent.parent.parent
    bundled = package_dir / "data" / "frcmod.modxna"
    if bundled.exists():
        return bundled

    return None


if __name__ == "__main__":
    main()
