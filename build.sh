#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT/docs"

# --- Parse arguments ---
# --current-only : skip building old SDK versions (faster for local testing)
# --local-pypi DIR : use a local PEP 503 Simple Repository (e.g. for unreleased packages)
BUILD_OLD_VERSIONS=true
LOCAL_PYPI=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --current-only) BUILD_OLD_VERSIONS=false; shift ;;
        --local-pypi)   LOCAL_PYPI="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Find the default SDK file ---
DEFAULT_SDK_FILE=""
for f in ../sdk*_default.txt; do
    [ -f "$f" ] && DEFAULT_SDK_FILE="$f" && break
done
[ -z "$DEFAULT_SDK_FILE" ] && { echo "Error: No sdk*_default.txt found"; exit 1; }
echo "Using default SDK file: $DEFAULT_SDK_FILE"

mkdir -p public temp

# Build extra-index arguments (used throughout the script)
# When a local PyPI is specified, add it as an extra index and tell uv to
# consider all indexes equally (unsafe-best-match), otherwise uv's default
# strategy ignores indexes once it finds the package name on another one.
UV_LOCAL_ARGS=""
if [ -n "$LOCAL_PYPI" ]; then
    LOCAL_INDEX_URL="file://$LOCAL_PYPI"
    UV_LOCAL_ARGS="--extra-index-url $LOCAL_INDEX_URL --index-strategy unsafe-best-match"
    echo "Using local PyPI: $LOCAL_PYPI"
fi

# Install doc build requirements
uv pip install pip packaging wheel setuptools "setuptools_scm<10"
uv pip install -r requirements.txt

# --- Helper functions ---

# Extract bare package name from a requirements line (strips extras and version specifiers)
pkg_name() { echo "$1" | sed 's/\[.*//;s/[<>=!~].*//'; }

is_external() { [[ "$1" =~ ^(qrisp)$ ]]; }

# Print non-external package names from an SDK file
sdk_packages() {
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        local name; name=$(pkg_name "$line")
        is_external "$name" || echo "$name"
    done < "$1"
}

# Track conf.py files that need restoring on exit
CONF_BACKUPS=()
cleanup_conf_backups() {
    for bak in "${CONF_BACKUPS[@]}"; do
        [ -f "$bak" ] && mv "$bak" "${bak%.bak}"
    done
    true  # ensure trap doesn't set a non-zero exit code
}
trap cleanup_conf_backups EXIT

# Build Sphinx docs for a single package source tree.
# Usage: build_sphinx <source-dir> <output-base-dir>
build_sphinx() {
    local src=$1 out_base=$2
    [ -f "$src/pyproject.toml" ] || { echo "Skip $src (no pyproject.toml)"; return 0; }
    [ -d "$src/docs" ]           || { echo "Skip $src (no docs/)"; return 0; }

    local name
    name=$(python -c "import tomllib; print(tomllib.load(open('$src/pyproject.toml','rb'))['project']['name'])")
    local out="$out_base/$name"
    echo "==> Building docs: $name"

    # Temporarily append shared intersphinx config
    cp "$src/docs/conf.py" "$src/docs/conf.py.bak"
    CONF_BACKUPS+=("$src/docs/conf.py.bak")
    cat "$REPO_ROOT/sphinx_docs_conf.py" >> "$src/docs/conf.py"

    (cd "$src" && python -m sphinx -j auto docs "$out")
    touch "$out/.nojekyll"

    # Restore original conf.py
    mv "$src/docs/conf.py.bak" "$src/docs/conf.py"
}

# Verify all expected packages were built
verify_packages() {
    local sdk_file=$1 out_dir=$2
    local missing=()
    for name in $(sdk_packages "$sdk_file"); do
        [ -d "$out_dir/$name" ] || missing+=("$name")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "❌ Missing packages in $out_dir: ${missing[*]}"
        return 1
    fi
    echo "✅ All packages built in $out_dir"
}

# ============================================================
# Download, install, and build docs for a given SDK file
# ============================================================
# Usage: build_version <sdk-file> <output-dir> <temp-subdir>
build_version() {
    local sdk_file=$1 out_dir=$2 tmp_dir=$3

    echo "Building documentation for $sdk_file into $out_dir"
    mkdir -p "$out_dir" "$tmp_dir"

    local filtered="$tmp_dir/filtered.txt"
    grep -v -E "^(qrisp)(\[|==|>=|<=|>|<|!=|~=|$)" "$sdk_file" > "$filtered"

    # Compile constraints and download sdists.
    if uv pip compile --upgrade --no-emit-index-url --no-emit-find-links \
        --no-header --no-cache --no-annotate \
        $UV_LOCAL_ARGS \
        --output-file "$tmp_dir/constraints.txt" "$filtered"; then
        USE_CONSTRAINTS=true
    else
        echo "WARNING: constraint compilation failed for $sdk_file; proceeding without constraints (see resolver error above)." >&2
        USE_CONSTRAINTS=false
    fi

    # Copy sdists from local PyPI directory (pip can't reliably use file:// PEP 503 indexes)
    if [ -n "$LOCAL_PYPI" ]; then
        while IFS= read -r line; do
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            local pkg; pkg=$(pkg_name "$line")
            local ver; ver=$(echo "$line" | grep -oE '==([0-9]+(\.[0-9]+)*)' | sed 's/==//')
            [ -z "$ver" ] && continue
            local pkg_under="${pkg//-/_}"
            local sdist="$LOCAL_PYPI/$pkg/${pkg_under}-${ver}.tar.gz"
            if [ -f "$sdist" ]; then
                echo "Copying local sdist: $sdist"
                cp "$sdist" "$tmp_dir/"
            fi
        done < "$filtered"
    fi

    # Download any remaining sdists from remote indexes
    # Build a requirements file excluding packages already obtained locally
    local remaining="$tmp_dir/remaining.txt"
    > "$remaining"
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        local pkg; pkg=$(pkg_name "$line")
        local pkg_under="${pkg//-/_}"
        # Check if we already have this package's sdist
        if ! ls "$tmp_dir"/${pkg_under}-*.tar.gz &>/dev/null; then
            echo "$line" >> "$remaining"
        fi
    done < "$filtered"

    if [ -s "$remaining" ]; then
        echo "Downloading remaining packages from remote indexes..."
        if [ "$USE_CONSTRAINTS" = true ]; then
            uv run -m pip download --no-deps --no-binary=:all: \
                -c "$tmp_dir/constraints.txt" -r "$remaining" -d "$tmp_dir"
        else
            uv run -m pip download --no-deps --no-binary=:all: \
                -r "$remaining" -d "$tmp_dir" || true
        fi
    fi

    # Install packages into venv (Sphinx needs them to resolve imports).
    if [ "$USE_CONSTRAINTS" = true ]; then
        uv pip install $UV_LOCAL_ARGS -r "$tmp_dir/constraints.txt"
    elif ! uv pip install $UV_LOCAL_ARGS -r "$filtered"; then
        echo "ERROR: failed to install packages from $sdk_file." >&2
        echo "       If an internal package index is configured (e.g. UV_EXTRA_INDEX_URL or" >&2
        echo "       PIP_EXTRA_INDEX_URL), it may be shadowing the versions pinned in the SDK" >&2
        echo "       file. Unset it before running, or use --local-pypi for unreleased packages." >&2
        exit 1
    fi

    # Extract and build each downloaded sdist
    for tarball in "$tmp_dir"/*.tar.gz; do
        [ -f "$tarball" ] || continue
        tar -xzf "$tarball" -C "$tmp_dir"
        local src_dir="$tmp_dir/$(basename "$tarball" .tar.gz)"
        [ -d "$src_dir" ] || continue
        build_sphinx "$src_dir" "$(cd "$out_dir" && pwd)"
    done
}

# ============================================================
# Build default/current version
# ============================================================

echo "=== Building default/current version ==="
build_version "$DEFAULT_SDK_FILE" "public" "temp/current"
verify_packages "$DEFAULT_SDK_FILE" "public"

# ============================================================
# Build older SDK versions
# ============================================================

if $BUILD_OLD_VERSIONS; then
    for sdk_file in ../sdk*.txt; do
        [ -f "$sdk_file" ] || continue
        [ "$sdk_file" = "$DEFAULT_SDK_FILE" ] && continue

        version=$(basename "$sdk_file" .txt)
        echo "=== Building $version ==="
        build_version "$sdk_file" "public/$version" "temp/$version"
        verify_packages "$sdk_file" "public/$version"
    done
fi

# ============================================================
# Cleanup and search index generation
# ============================================================

rm -rf public/jupyter_execute
find public -type d -name .doctrees -exec rm -rf {} +
touch public/.nojekyll

echo "Generating search index for main version..."
python generate_search_index.py
cp search.json public/ 2>/dev/null || echo "No search index found"

for version_dir in public/sdk*; do
    [ -d "$version_dir" ] || continue
    version_name=$(basename "$version_dir")
    echo "Generating search index for $version_name..."
    python generate_search_index.py "$version_name" "$version_dir/" "search_${version_name}.json"
    cp "search_${version_name}.json" "$version_dir/" 2>/dev/null || true
    rm -f "search_${version_name}.json"
done

echo "Done."
