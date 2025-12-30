"""Export MD trajectory as video using MDAnalysis and nglview."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def export_video(
    topology: Path,
    trajectory: Path,
    output: Path,
    fps: int = 30,
    stride: int = 1,
    width: int = 800,
    height: int = 600,
):
    """Export trajectory as MP4 video.

    Uses cpptraj to process trajectory and nglview for rendering.

    Args:
        topology: Path to AMBER topology (.parm7)
        trajectory: Path to trajectory file (.nc)
        output: Output video path (.mp4)
        fps: Frames per second (default: 30)
        stride: Skip every N frames (default: 1, use all frames)
        width: Video width in pixels
        height: Video height in pixels
    """
    import MDAnalysis as mda
    from MDAnalysis.analysis import align

    print(f"Loading trajectory: {trajectory}")
    print(f"Using topology: {topology}")

    # Load trajectory
    u = mda.Universe(str(topology), str(trajectory))
    n_frames = len(u.trajectory)
    print(f"Loaded {n_frames} frames")

    # Select non-water atoms for alignment and visualization
    oligo = u.select_atoms("not (resname WAT or resname Na+ or resname Cl-)")
    print(f"Selected {len(oligo)} atoms (excluding solvent)")

    if len(oligo) == 0:
        print("Warning: No non-solvent atoms found. Using all atoms.")
        oligo = u.atoms

    # Align trajectory to first frame
    print("Aligning trajectory to first frame...")
    align.AlignTraj(u, u, select="not (resname WAT or resname Na+ or resname Cl-)", in_memory=True).run()

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_file = tmpdir / "trajectory.pdb"

        # Export aligned trajectory as multi-frame PDB (stripped of water)
        print(f"Exporting frames (stride={stride})...")
        with mda.Writer(str(pdb_file), oligo.n_atoms) as writer:
            for i, ts in enumerate(u.trajectory[::stride]):
                writer.write(oligo)

        n_output_frames = (n_frames + stride - 1) // stride
        print(f"Wrote {n_output_frames} frames to PDB")

        # Render frames using nglview
        frames_dir = tmpdir / "frames"
        frames_dir.mkdir()

        print("Rendering frames with nglview...")
        _render_frames_nglview(pdb_file, frames_dir, width, height)

        # Compile video with ffmpeg
        print(f"Compiling video at {fps} fps...")
        _compile_video(frames_dir, output, fps)

    print(f"Video saved to: {output}")


def _render_frames_nglview(pdb_file: Path, output_dir: Path, width: int, height: int):
    """Render PDB frames to PNG images using nglview."""
    import nglview as nv
    from PIL import Image
    import io

    # Load multi-model PDB
    view = nv.show_file(str(pdb_file))

    # Configure representation
    view.clear_representations()
    view.add_representation("cartoon", selection="nucleic", color="chainindex")
    view.add_representation("licorice", selection="nucleic and not backbone", color="element")

    # Set background and camera
    view.background = "white"
    view._set_size(f"{width}px", f"{height}px")

    # Render each frame
    n_frames = view.max_frame + 1
    print(f"  Rendering {n_frames} frames...")

    for i in range(n_frames):
        view.frame = i
        # Get image data
        image_data = view.render_image()
        if image_data:
            img = Image.open(io.BytesIO(image_data))
            img.save(output_dir / f"frame_{i:06d}.png")

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{n_frames} frames")


def _render_frames_simple(pdb_file: Path, output_dir: Path, width: int, height: int):
    """Simple fallback renderer using matplotlib 3D projection."""
    import MDAnalysis as mda
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    u = mda.Universe(str(pdb_file))
    n_frames = len(u.trajectory)

    print(f"  Rendering {n_frames} frames with matplotlib...")

    for i, ts in enumerate(u.trajectory):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        pos = u.atoms.positions
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, c="steelblue", alpha=0.7)

        # Set equal aspect ratio
        max_range = np.ptp(pos, axis=0).max() / 2
        mid = pos.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title(f"Frame {i}")

        plt.savefig(output_dir / f"frame_{i:06d}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i + 1}/{n_frames} frames")


def _compile_video(frames_dir: Path, output: Path, fps: int):
    """Compile PNG frames to MP4 using ffmpeg."""
    # Find frame files
    frame_pattern = frames_dir / "frame_%06d.png"

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", str(frame_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",  # Quality (lower = better, 18-28 is reasonable)
        str(output),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        raise RuntimeError("Failed to compile video with ffmpeg")


def main():
    """CLI entry point for video export."""
    parser = argparse.ArgumentParser(
        description="Export MD trajectory as video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("topology", type=Path, help="AMBER topology file (.parm7)")
    parser.add_argument("trajectory", type=Path, help="Trajectory file (.nc)")
    parser.add_argument("-o", "--output", type=Path, default=Path("trajectory.mp4"),
                        help="Output video path (default: trajectory.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame (default: 1)")
    parser.add_argument("--width", type=int, default=800, help="Video width (default: 800)")
    parser.add_argument("--height", type=int, default=600, help="Video height (default: 600)")
    parser.add_argument("--simple", action="store_true",
                        help="Use simple matplotlib renderer instead of nglview")

    args = parser.parse_args()

    export_video(
        topology=args.topology,
        trajectory=args.trajectory,
        output=args.output,
        fps=args.fps,
        stride=args.stride,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
