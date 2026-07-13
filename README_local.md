# Building the IQM SDK docs locally

These instructions reproduce, on your own machine, what the
`Build and Deploy Documentation` GitHub Actions workflow
(`.github/workflows/publish.yml`) does — everything except the final
`gh-pages` deploy step, which only runs on push to `main`.

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv) — Python environment & installer
- Node.js (CI uses **v18**) + `npm` — for the React front page
- [`graphviz`](https://graphviz.org/download/) — optional; some diagrams won't
  render without the `dot` binary

## ⚠️ Watch out for internal package indexes

CI runs in a clean environment that installs everything from public PyPI. If
your shell has an internal index configured, it can **shadow** the versions
pinned in the `sdk*.txt` files and break the build with a confusing
`Invalid version: 'unknown'` error deep inside Sphinx.

Check for these before building:

```bash
env | grep -iE 'UV_INDEX|UV_EXTRA_INDEX|PIP_INDEX|PIP_EXTRA_INDEX'
```

If anything is set, run the build with those
variables unset so your environment matches CI:

## 1. Create the Python environment

Match CI's Python version:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

## 2. Build the package documentation + search index

```bash
uv pip install setuptools        # CI installs this before building
bash build.sh                    # add --current-only to skip old SDK versions (faster)
```

`build.sh`:

- reads the default SDK file (`sdk*_default.txt`) plus any older `sdk*.txt`,
- downloads and installs the pinned package versions,
- builds each package's Sphinx docs into `docs/public/<package-name>/`,
- generates the search index (`search.json`).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--current-only` | Build only the default SDK version (skip older `sdk*.txt`) — much faster for local iteration. |
| `--local-pypi DIR` | Resolve packages from a local PEP 503 index (for unreleased packages). |

## 3. Build the React front page

```bash
cd docs
./copy-sdk-files.sh              # copies ../sdk*.txt into docs/public

# Version range shown in the UI is derived from the sdk*.txt filenames.
# With sdk4_5_default.txt present, max version is 4.5:
export VITE_SDK_MAX_MAJOR_VERSION=4
export VITE_SDK_MAX_MINOR_VERSION=5

npm ci
npm run build

mkdir -p public
cp -r dist/* public/
cp src/favicon.ico public/favicon.ico
```

## 4. Serve it

```bash
cd docs/public
python3 -m http.server 8000
# open http://localhost:8000
```

## Faster front-end-only workflow

If you're only changing the React front page and don't need freshly built
package docs, skip `build.sh`. Download `search.json` from the
[`gh-pages` branch](https://github.com/iqm-finland/docs/tree/gh-pages), drop it
into `docs/`, then:

```bash
cd docs
npm install
npm run dev          # Vite dev server with hot reload
```

(Yarn users: `yarn install && yarn dev`.)

## Adding a new OS version

Version configs are auto-generated from `sdkX_Y.txt` files in the repo root
(the `_default.txt` suffix marks the default version). See the
"Adding New OS Versions" section of `docs/README.md` for details.
