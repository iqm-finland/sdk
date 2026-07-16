# IQM SDK Docs

## Content of this repository
This repository contains the documentation for the IQM SDK. The documentation
is hosted on GitHub Pages https://docs.iqm.tech/.

Note that public IQM Resonance quantum computers do not always support the
latest versions of client packages.

Refer to the Resonance user guides and documentation to find the compatible
versions for different quantum computers.

This GitHub repository is a read-only mirror that isn't used for accepting
contributions.


## How to obtains package sources

To obtain the source code of the IQM SDK packages, please visit the
[IQM SDK PyPI page](https://pypi.org/search/?q=iqm) and download the source
distribution (sdist) for the desired package and version.

An alternative way to obtain the source code is to use curl and extract the
source distribution directly from PyPI. For example, to download and extract
 the source code for the `iqm-client` package version 35.0.0, you can use the
 following command:

```bash
curl -s https://pypi.org/pypi/iqm-client/35.0.0/json | jq -r '.urls[] | select(.packagetype == "sdist") | .url'
```

**For support, contact `support@iqm.tech`**.
