#!/bin/bash
# install.sh - Install modXNA pipeline with isolated conda environment
# Supports: macOS (arm64/x86_64), Linux (x86_64)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$BASE_DIR/bin"

echo "=== modXNA Pipeline Installation ==="
echo "Base directory: $BASE_DIR"
echo "Bin directory: $BIN_DIR"

# Create bin directory if it doesn't exist
mkdir -p "$BIN_DIR"

# Detect OS and architecture
detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Darwin)
            case "$arch" in
                arm64)
                    echo "osx-arm64"
                    ;;
                x86_64)
                    echo "osx-64"
                    ;;
                *)
                    echo "Unsupported macOS architecture: $arch" >&2
                    exit 1
                    ;;
            esac
            ;;
        Linux)
            case "$arch" in
                x86_64)
                    echo "linux-64"
                    ;;
                *)
                    echo "Unsupported Linux architecture: $arch" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "Unsupported operating system: $os" >&2
            exit 1
            ;;
    esac
}

PLATFORM=$(detect_platform)
echo "Detected platform: $PLATFORM"

# Download micromamba
MICROMAMBA_BIN="$BIN_DIR/micromamba"
if [ ! -f "$MICROMAMBA_BIN" ]; then
    echo ""
    echo "=== Downloading micromamba ==="
    MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/$PLATFORM/latest"
    echo "URL: $MICROMAMBA_URL"

    # Download and extract micromamba
    curl -Ls "$MICROMAMBA_URL" | tar -xvj -C "$BIN_DIR" --strip-components=1 bin/micromamba
    chmod +x "$MICROMAMBA_BIN"
    echo "Micromamba installed to: $MICROMAMBA_BIN"
else
    echo "Micromamba already exists at: $MICROMAMBA_BIN"
fi

# Create isolated conda environment
CONDA_ENV_DIR="$BIN_DIR/conda_env"
if [ ! -d "$CONDA_ENV_DIR" ]; then
    echo ""
    echo "=== Creating isolated conda environment ==="
    "$MICROMAMBA_BIN" create -y -p "$CONDA_ENV_DIR" -c conda-forge python=3.11
    echo "Conda environment created at: $CONDA_ENV_DIR"
else
    echo "Conda environment already exists at: $CONDA_ENV_DIR"
fi

# Install ambertools and build dependencies
echo ""
echo "=== Installing ambertools and build dependencies from conda-forge ==="
"$MICROMAMBA_BIN" install -y -p "$CONDA_ENV_DIR" -c conda-forge \
    ambertools \
    'numpy<2' \
    netcdf4 fftw cmake make gfortran
echo "Ambertools and dependencies installed successfully"

# Clone modXNA repository
MODXNA_DIR="$BIN_DIR/modXNA"
if [ ! -d "$MODXNA_DIR" ]; then
    echo ""
    echo "=== Cloning modXNA repository ==="
    git clone https://github.com/drroe/modXNA.git "$MODXNA_DIR"
    echo "modXNA cloned to: $MODXNA_DIR"
else
    echo "modXNA already exists at: $MODXNA_DIR"
    echo "Updating modXNA repository..."
    (cd "$MODXNA_DIR" && git pull)
fi

# Build cpptraj from source (conda-forge version is too old for modXNA)
CPPTRAJ_BUILD_DIR="$BIN_DIR/cpptraj_build"
CPPTRAJ_INSTALL_DIR="$BIN_DIR/cpptraj_install"
if [ ! -f "$CPPTRAJ_INSTALL_DIR/bin/cpptraj" ]; then
    echo ""
    echo "=== Building cpptraj from source ==="
    echo "Note: conda-forge cpptraj (6.24) is too old; modXNA requires >= 6.26"

    # Clone if not present
    if [ ! -d "$CPPTRAJ_BUILD_DIR" ]; then
        git clone --depth 1 https://github.com/Amber-MD/cpptraj.git "$CPPTRAJ_BUILD_DIR"
    fi

    # Build with cmake
    mkdir -p "$CPPTRAJ_BUILD_DIR/build"
    cd "$CPPTRAJ_BUILD_DIR/build"

    # Set up PATH for build
    export PATH="$CONDA_ENV_DIR/bin:/usr/bin:/bin:$PATH"

    # Configure with system clang (avoids conda linker issues on macOS)
    CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. \
        -DCOMPILER=CLANG \
        -DCMAKE_INSTALL_PREFIX="$CPPTRAJ_INSTALL_DIR"

    # Build and install
    make -j4
    make install

    cd "$BASE_DIR"
    echo "cpptraj built and installed to: $CPPTRAJ_INSTALL_DIR"
else
    echo "cpptraj already installed at: $CPPTRAJ_INSTALL_DIR"
fi

# Generate setup_env.sh
SETUP_ENV_SCRIPT="$SCRIPT_DIR/setup_env.sh"
echo ""
echo "=== Generating setup_env.sh ==="

cat > "$SETUP_ENV_SCRIPT" << 'SETUP_EOF'
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
SETUP_EOF

chmod +x "$SETUP_ENV_SCRIPT"
echo "Generated: $SETUP_ENV_SCRIPT"

# Make scripts executable
chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To use the modXNA pipeline, source the setup script:"
echo "  source $SETUP_ENV_SCRIPT"
echo ""
echo "Then run the pipeline with:"
echo "  uv run python -m modxna.cli 'RNA1{[moe](C)[sp].[moe](T)[sp].d(A)}\$\$\$\$' -o output/"
echo ""
echo "Installed components:"
echo "  - Micromamba:  $MICROMAMBA_BIN"
echo "  - Conda env:   $CONDA_ENV_DIR"
echo "  - Ambertools:  installed in conda env"
echo "  - cpptraj:     $CPPTRAJ_INSTALL_DIR (v6.30+)"
echo "  - modXNA:      $MODXNA_DIR"
