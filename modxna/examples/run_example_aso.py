#!/usr/bin/env python3
"""Run a complete MD simulation of the example ASO and export video.

This script demonstrates the full modXNA workflow:
1. Build system from HELM notation
2. Run MD simulation on CPU
3. Export trajectory video

Usage:
    # From modxna directory:
    uv run python examples/run_example_aso.py

    # Or with custom settings:
    uv run python examples/run_example_aso.py --simulation-ns 1 --threads 16
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run example ASO simulation and export video"
    )
    parser.add_argument(
        "--simulation-ns",
        type=float,
        default=1.0,
        help="Simulation length in ns (default: 1)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of CPU threads (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_output"),
        help="Output directory (default: example_output)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip system building (use existing files)",
    )
    parser.add_argument(
        "--skip-md",
        action="store_true",
        help="Skip MD simulation (use existing trajectory)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video export",
    )

    args = parser.parse_args()

    # Get paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    helm_file = script_dir / "example_aso.helm"
    output_dir = args.output_dir.resolve()

    print("=" * 60)
    print("modXNA Example: ASO MD Simulation")
    print("=" * 60)
    print(f"HELM file:      {helm_file}")
    print(f"Output dir:     {output_dir}")
    print(f"Simulation:     {args.simulation_ns} ns")
    print(f"CPU threads:    {args.threads}")
    print("=" * 60)

    # Step 1: Build system
    if not args.skip_build:
        print("\n[Step 1/3] Building system from HELM...")
        cmd = [
            sys.executable, "-m", "modxna.cli",
            str(helm_file),
            "-o", str(output_dir),
            "--simulation-ns", str(args.simulation_ns),
            "--threads", str(args.threads),
            "-v",
        ]
        result = subprocess.run(cmd, cwd=project_dir)
        if result.returncode != 0:
            print("Error: System build failed")
            sys.exit(1)
    else:
        print("\n[Step 1/3] Skipping system build (--skip-build)")

    # Step 2: Run MD simulation
    md_dir = output_dir / "md_inputs"
    if not args.skip_md:
        print("\n[Step 2/3] Running MD simulation...")
        print(f"  This may take 1-2 hours for {args.simulation_ns} ns on CPU...")
        print(f"  Working directory: {md_dir}")

        # Source environment and run
        run_script = md_dir / "run_md.sh"
        if not run_script.exists():
            print(f"Error: run_md.sh not found at {run_script}")
            sys.exit(1)

        # Run the simulation
        result = subprocess.run(
            ["bash", str(run_script)],
            cwd=md_dir,
            env={
                **subprocess.os.environ,
                "OMP_NUM_THREADS": str(args.threads),
            },
        )
        if result.returncode != 0:
            print("Error: MD simulation failed")
            sys.exit(1)
    else:
        print("\n[Step 2/3] Skipping MD simulation (--skip-md)")

    # Step 3: Export video
    trajectory = md_dir / "trajectories" / "prod.nc"
    topology = output_dir / "aso.parm7"
    video_output = output_dir / f"aso_{args.simulation_ns}ns.mp4"

    if not args.skip_video:
        print("\n[Step 3/3] Exporting trajectory video...")

        if not trajectory.exists():
            print(f"Error: Trajectory not found at {trajectory}")
            print("  Run the MD simulation first (remove --skip-md)")
            sys.exit(1)

        cmd = [
            sys.executable, "-m", "modxna.video_export",
            str(topology),
            str(trajectory),
            "-o", str(video_output),
            "--fps", "30",
            "--stride", "10",  # Use every 10th frame for reasonable video length
        ]
        result = subprocess.run(cmd, cwd=project_dir)
        if result.returncode != 0:
            print("Error: Video export failed")
            sys.exit(1)
    else:
        print("\n[Step 3/3] Skipping video export (--skip-video)")

    # Summary
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    if not args.skip_md and trajectory.exists():
        print(f"Trajectory:       {trajectory}")
    if not args.skip_video and video_output.exists():
        print(f"Video:            {video_output}")


if __name__ == "__main__":
    main()
