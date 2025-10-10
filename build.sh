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

# Function to build documentation for a given SDK file and output directory
build_docs() {
    local SDK_FILE=$1
    local OUTPUT_DIR=$2
    local TEMP_SUBDIR=$3
    
    echo "Building documentation for $SDK_FILE into $OUTPUT_DIR"
    
    # Create subdirectories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "temp/$TEMP_SUBDIR"
    
    # Download and extract the packages' source distributions
    uv run -m pip download --no-deps --no-binary=:all: -r "$SDK_FILE" -d "./temp/$TEMP_SUBDIR"
    
    # Install the packages into the virtual environment; needed for Sphinx to resolve namespaces
    uv pip install -r "$SDK_FILE"
    
    echo "Downloaded packages for $SDK_FILE:"
    ls -la "temp/$TEMP_SUBDIR"
    
    # Iterate over the downloaded source distributions
    for SDIST_FILE in "./temp/$TEMP_SUBDIR"/*.tar.gz; do
        # Skip if no files match the pattern
        [ -f "$SDIST_FILE" ] || continue
        
        # Extract the package sdist to temp directory
        echo "Extracting $SDIST_FILE..."
        tar -xzf "$SDIST_FILE" -C "./temp/$TEMP_SUBDIR"
        
        # Get the extracted directory name (remove .tar.gz and path)
        BASENAME=$(basename "$SDIST_FILE" .tar.gz)
        SRC_DIR="./temp/$TEMP_SUBDIR/$BASENAME"
        echo "Extracted to ${SRC_DIR}"
        
        # Check if the extracted directory exists
        if [ ! -d "$SRC_DIR" ]; then
            echo "Error: Extracted directory $SRC_DIR does not exist"
            continue
        fi
        
        # Go to the package source directory
        cd "$SRC_DIR"
        
        # Check if pyproject.toml exists
        if [ ! -f "pyproject.toml" ]; then
            echo "Error: pyproject.toml not found in $SRC_DIR"
            cd ../../..
            continue
        fi
        
        # Get the package name from the pyproject.toml file
        PKG_NAME=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['name'])")
        
        # Check if docs directory exists
        if [ ! -d "docs" ]; then
            echo "Warning: docs directory not found in $SRC_DIR, skipping..."
            cd ../../..
            continue
        fi
        
        # Build the docs and save to the specified output directory
        # Calculate the relative path back to the docs directory based on nesting level
        if [[ "$TEMP_SUBDIR" == "current" ]]; then
            RELATIVE_PATH="../../../"
        else
            RELATIVE_PATH="../../../../"
        fi
        
        OUTPUT_PATH="../../../$OUTPUT_DIR/${PKG_NAME%[*}"
        cat "${RELATIVE_PATH}sphinx_docs_conf.py" >> docs/conf.py
        python -m sphinx docs "$OUTPUT_PATH"
        # add .nojekyll in order to stop Github from treating the directory as a Jekyll blog generator,
        # which ignores directories starting with underscore
        touch "$OUTPUT_PATH/.nojekyll"
        
        # Go back to the docs directory
        cd ../../..
    done
}

# Build current versions from sdk.txt
build_docs "../sdk.txt" "public" "current"

# Build older versions from all SDK version files (excluding the main sdk.txt)
for sdk_file in ../sdk*.txt; do
    if [ -f "$sdk_file" ]; then
        # Skip the main sdk.txt file
        if [ "$(basename "$sdk_file")" = "sdk.txt" ]; then
            continue
        fi
        
        # Extract version from filename (e.g., sdk4_1.txt -> sdk4_1, sdk5_1.txt -> sdk5_1)
        version=$(basename "$sdk_file" .txt)
        echo "Building documentation for $version from $sdk_file"
        build_docs "$sdk_file" "public/$version" "$version"
    fi
done

# Remove the Jupyter notebook execution directory and pickled doctree caches
rm -rf public/jupyter_execute
find public -type d -name .doctrees -exec rm -rf {} +

# add .nojekyll in order to stop Github from treating the directory as a Jekyll blog generator,
# which ignores directories starting with underscore
touch public/.nojekyll

# Generate search index for main/current version
echo "Generating search index for main version..."
python generate_search_index.py
cp search.json public/ || echo "No search index found"

# Generate search indices for each SDK version
for version_dir in public/sdk*; do
    if [ -d "$version_dir" ]; then
        version_name=$(basename "$version_dir")
        echo "Generating search index for $version_name..."
        
        # Use the generate_search_index.py script for versioned documentation
        python generate_search_index.py "$version_name" "$version_dir/" "search_${version_name}.json"
        
        # Copy the search index to the version directory
        cp "search_${version_name}.json" "$version_dir/" || echo "No search index found for $version_name"
        
        # Clean up temporary file
        rm "search_${version_name}.json" 2>/dev/null || true
    fi
done
