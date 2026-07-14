# Building the IQM SDK docs locally

These instructions reproduce, on your own machine, what the GitHub Actions workflow
(`.github/workflows/publish.yml`) does — everything except the final deployment.


# Building with Mise

[Mise](https://mise.jdx.dev/) is a tool that automates the building of the SDK docs.
It handles the Python environment, package installation, Sphinx build, site building
and HTTP server for you.

To install necessary tools (`uv`, `node`, `python`), run:
```bash
mise list # To see what tools are installed
mise install
```

Separate tasks can be listed with
```bash
mise tasks
```

Mise can be allowed to run all necessary tasks with one command or you can run each
task separately.

All in one command:
```bash
mise run http-serve-docs
```

One task at a time:
```bash
mise run build-sphinx
mise run package-docs
mise run serve-http
```

# Building without Mise

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv) — Python environment & installer
- Node.js (v24 has been tested) + `npm` — for the React front page
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
variables unset.

## 1. Create the Python environment

Match CI's Python version:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

## 2. Build the package documentation + search index

```bash
bash build.sh  # add --current-only to skip old SDK versions (faster)
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

npm ci
npm run build

mkdir -p public
cp -r dist/* public/
cp src/favicon.ico public/favicon.ico
```

## 4. Serve it

```bash
cd docs/public
python3 -m http.server --bind 127.0.0.1 8000
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
