# Building the documentation locally

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv)
- Node.js
- [`graphviz`](https://graphviz.org/download/) (optional; some diagrams will not be rendered without it)

## Info

`build.sh` is a script that is used by Github actions to build the documentation for each package, and compile a common search index.
Namely, it performs the following actions:

- download source distributions of packages specified in `../sdk.txt`
- install the packages specified in `../sdk.txt` into the current environment
- for each source distribution:
    - extract the archive
    - determine the package name by parsing its `pyproject.toml`
    - build documentation by calling `sphinx` and saves it into `public/<PACKAGE_NAME>`
- call `python generate_search_index.py` to generate the search index file `search.json`

Once packages' documentation directories and the search index file are in place, the front page (single-page app) can be built.

## Instructions

Create a Python environment with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Build the documentation for each package, and compile a common search index:

```bash
chmod +x build.sh
./build.sh
```

Build the front page for production use:

```bash
npm ci
npm run build
cp -r dist/* public/
cp src/favicon.ico public/favicon.ico
```

Then you can `cd` into `public` and serve the site with any web server, e.g. `python3 -m http.server 8000`, then open http://localhost:8000..

Alternatively, build the site locally in development mode:

```bash
npm install
npm run dev
```

(if you use Yarn: `yarn install && yarn dev`)

If you only need to work on the main React-powered page, no need to run `build.sh`; instead, just download `search.json` file from the [GitHub Pages branch](https://github.com/iqm-finland/docs/tree/gh-pages), put it in `./docs` and run `npm install && npm run dev`.


# Adding New OS Versions

The SDK documentation website now automatically detects and creates version configurations from SDK files! 🎉

## How It Works:

The system scans for `sdkX_Y.txt` files in the project root and automatically generates version configurations:

- **Pattern**: `sdkX_Y.txt` creates version `X_Y`
- **Default Version**: Files ending with `_default.txt` (e.g., `sdk4_3_default.txt`) become the default version
- **Path Mapping**: Default version maps to `/`, others to `/sdkX_Y/`
- **Automatic Sorting**: Versions are sorted by version number (newest first: 4_4, 4_3, 4_1, 4_0, 3_4)
- **Preview Warnings**: Versions newer than the default show preview warnings

## To Add a New Version:

1. **Create the SDK file** in the project root:
   ```
   # Example: sdk5_0.txt
   iqm-pulla[qiskit,qir]==12.0
   iqm-client[qiskit,cirq,cli]==35.0
   iqm-pulse==15.0
   iqm-data-definitions==8.0
   ```

2. **For a new default version** (optional):
   ```
   # Rename existing default and create new one
   mv sdk4_3_default.txt sdk4_3.txt
   cp sdk5_0.txt sdk5_0_default.txt
   ```

That's it! The version will automatically appear in the UI with:
- ✅ Proper sorting (newest versions on the left)
- ✅ Automatic path mapping 
- ✅ Package detection from SDK file
- ✅ Preview warnings for versions newer than default
- ✅ URL persistence (e.g., `?version=5_0`)

## SDK File Format:

Each line should contain a package name, optionally with extras and versions:
```
package-name
package-with-extras[extra1,extra2]
package-with-version==1.2.3
package-with-extras-and-version[extras]==1.0.0
```

## Configuration Properties (Auto-Generated):

- **id**: Version identifier (e.g., `4_3`, `5_0`)
- **label**: Display name (e.g., `IQM OS 4.3 (Resonance)`)
- **pathPrefix**: URL path (`./ for default, ./sdkX_Y/ for others`)
- **packages**: Extracted from SDK file content
- **isDefault**: `true` for `_default.txt` files
- **isPreview**: `true` for versions newer than default
- **description**: Auto-generated warnings for preview/older versions

## Manual Override (Fallback):

If automatic detection fails, the system falls back to static configuration in `src/configs.ts`.