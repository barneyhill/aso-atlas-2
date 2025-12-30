"""Build modXNA residue library files."""

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .modxna_mapper import ModXNAResidue


def _get_modxna_env(modxna_path: Path) -> dict:
    """Build environment dict with paths for modXNA and AmberTools.

    Args:
        modxna_path: Path to modXNA installation

    Returns:
        Environment dict with updated PATH and AMBERHOME
    """
    env = os.environ.copy()

    # modXNA is in bin/modXNA, conda_env is in bin/conda_env
    bin_dir = modxna_path.parent  # Go from bin/modXNA to bin/
    conda_bin = bin_dir / "conda_env" / "bin"
    cpptraj_bin = bin_dir / "cpptraj_install" / "bin"

    # Update PATH to include cpptraj_install/bin, conda_env/bin and modXNA
    # cpptraj_install has newer cpptraj (6.30) required by modXNA
    path_parts = [
        str(cpptraj_bin),
        str(conda_bin),
        str(modxna_path),
        env.get("PATH", ""),
    ]
    env["PATH"] = ":".join(path_parts)

    # Set AMBERHOME for AmberTools
    env["AMBERHOME"] = str(bin_dir / "conda_env")

    return env


def build_residue(residue: "ModXNAResidue", output_dir: Path, modxna_path: Path) -> Path:
    """Build a single residue using modXNA.sh.

    Args:
        residue: ModXNAResidue object containing backbone, sugar, base, resname, and cap info
        output_dir: Base output directory
        modxna_path: Path to modXNA installation (containing modXNA.sh)

    Returns:
        Path to the generated .lib file
    """
    # 1. Create output_dir/residues/RESNAME/
    residue_dir = output_dir / "residues" / residue.resname
    residue_dir.mkdir(parents=True, exist_ok=True)

    # 2. Write RESNAME.in.modxna with: BACKBONE SUGAR BASE
    input_file = residue_dir / f"{residue.resname}.in.modxna"
    with open(input_file, "w") as f:
        f.write(f"{residue.backbone} {residue.sugar} {residue.base}\n")

    # 3. Build modXNA command
    modxna_sh = modxna_path / "modxna.sh"
    cmd = [
        str(modxna_sh),
        "-i", str(input_file),
        "-m", residue.resname
    ]

    # Add cap flags if needed
    if residue.is_5prime:
        cmd.append("--5cap")
    if residue.is_3prime:
        cmd.append("--3cap")

    # 4. Run modXNA.sh with proper environment
    env = _get_modxna_env(modxna_path)
    try:
        result = subprocess.run(
            cmd,
            cwd=residue_dir,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"modXNA failed for {residue.resname}:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        )

    # 5. Return path to .lib file
    lib_file = residue_dir / f"{residue.resname}.lib"
    if not lib_file.exists():
        # Try alternate naming conventions
        lib_file = residue_dir / f"{residue.resname}.off"
        if not lib_file.exists():
            # Look for any .lib file in the directory
            lib_files = list(residue_dir.glob("*.lib"))
            if lib_files:
                lib_file = lib_files[0]
            else:
                raise FileNotFoundError(
                    f"No .lib file generated for {residue.resname} in {residue_dir}"
                )

    return lib_file


def build_all_residues(
    residues: list["ModXNAResidue"],
    output_dir: Path,
    modxna_path: Path
) -> dict[str, Path]:
    """Build all unique residues, return mapping of resname -> lib_path.

    Args:
        residues: List of ModXNAResidue objects to build
        output_dir: Base output directory
        modxna_path: Path to modXNA installation

    Returns:
        Dictionary mapping residue names to their .lib file paths
    """
    lib_paths: dict[str, Path] = {}

    # Get unique residues by resname (already unique from get_unique_residues)
    seen_resnames: set[str] = set()

    for residue in residues:
        if residue.resname in seen_resnames:
            continue
        seen_resnames.add(residue.resname)

        print(f"Building residue: {residue.resname} "
              f"({residue.backbone} {residue.sugar} {residue.base})")

        lib_path = build_residue(residue, output_dir, modxna_path)
        lib_paths[residue.resname] = lib_path

        print(f"  -> {lib_path}")

    return lib_paths
