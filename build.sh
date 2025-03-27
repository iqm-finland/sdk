#!/usr/bin/env bash

cd docs

# Put package names into environment variable; later used for creating directories and for the React app
PACKAGES=$(awk '{print $1}' ../sdk.txt | tr '\n' ' ' | sed 's/ $//')

echo "Building documentation for packages: $PACKAGES"

# Public directory for the final site build
mkdir -p public
# Temporary directory for downloading and extracting the packages
mkdir -p temp

# Install the requirements for building the docs
uv pip install pip packaging wheel
uv pip install -r requirements.txt

# Download and extract the packages' source distributions
uv run -m pip download --no-deps --no-binary=:all: -r ../sdk.txt -d ./temp

# Install the packages into the virtual environment; needed for Sphinx to resolve namespaces
uv pip install -r ../sdk.txt

echo "Downloaded packages:"
ls -la temp

# Iterate over the downloaded source distributions
for SDIST_FILE in ./temp/*.tar.gz; do
    # Extract the package sdist to temp directory
    echo "Extracting $SDIST_FILE..."
    tar -xzf "$SDIST_FILE" -C ./temp

    SRC_DIR="${SDIST_FILE%.tar.gz}"
    echo "Extracted to ${SRC_DIR}"

    # Go to the package source directory
    cd "$SRC_DIR"
    # Get the package name from the pyproject.toml file
    PKG_NAME=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['name'])")

    # Build the docs and save to the public directory
    # we are now in ROOT/temp/<package_name>/ and public dir is ROOT/public/
    USE_LOCAL_TARGET=true python -m sphinx docs ../../public/${PKG_NAME%[*}
    # add .nojekyll in order to stop Github from treating the directory as a Jekyll blog generator,
    # which ignores directories starting with underscore
    touch ../../public/${PKG_NAME%[*}/.nojekyll

    # Go back to the docs directory
    cd ../..
done

# Remove the Jupyter notebook execution directory
rm -rf public/jupyter_execute
# add .nojekyll in order to stop Github from treating the directory as a Jekyll blog generator,
# which ignores directories starting with underscore
touch public/.nojekyll

uv run generate_search_index.py
cp search.json public/ || echo "No search index found"
