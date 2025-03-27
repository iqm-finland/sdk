# Building the documentation locally

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv)
- Node.js

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
