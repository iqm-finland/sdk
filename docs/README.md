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

To add a new OS version (e.g., IQM OS 5.0), you only need to edit one file: `docs/src/configs.ts`

## Steps:

1. **Update the VersionType**:
   ```typescript
   export type VersionType = 'resonance' | 'os4.1' | 'os4.2' | 'os5.0';
   ```

2. **Add the new version configuration**:
   ```typescript
   {
     id: 'os5.0',
     label: 'IQM OS 5.0',
     pathPrefix: './sdk5_0/',
     description: 'You are viewing documentation for IQM OS 5.0. Some packages may not be available in this version.',
     packages: [
       'iqm-pulla',
       'iqm-client',
       'iqm-pulse',
       // Add other packages available in OS 5.0
     ]
   }
   ```

3. **Create the SDK file** (for the build process):
   Create `sdk5_0.txt` in the root with the package versions:
   ```
   iqm-pulla[qiskit,qir]==12.0
   iqm-client[qiskit,cirq,cli]==35.0
   iqm-pulse==15.0
   ```

That's it! The build system, UI, and URL handling will automatically support the new version.

## Configuration Properties:

- **id**: Unique identifier used in URLs and internal logic
- **label**: Display name shown in the UI
- **pathPrefix**: URL path where the documentation will be served
- **description**: Optional warning/info message shown when this version is selected
- **packages**: Array of package names available in this version

## Automatic Features:

- Version selector buttons are generated automatically
- URL state persistence (e.g., `?version=os5.0`)
- Package filtering (only shows packages available in the selected version)
- Build system automatically processes `sdk*_*.txt` files
- Warning messages for non-default versions