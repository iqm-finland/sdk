# IQM Docs

This repository exists solely for building and publishing documentation pages for IQM's client-side libraries: [https://docs.meetiqm.com/](https://docs.meetiqm.com/)


# Building the documentation locally
It uses a Python script in `generate_search_index.py` to generate a search index for the documentation. To build the documentation locally, you need to have Python installed and then run the script based on the folders, or retrieve a `search.json` document from the GitHub repository and the [GitHub Pages branch](https://github.com/iqm-finland/docs/tree/gh-pages).

In order to locally run the React application, you need to have Node.js installed. Then you can run the following commands:

```bash
npm install
npm run dev
```

or use Yarn:
```bash
yarn install
yarn  dev
```

