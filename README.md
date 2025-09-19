# IQM SDK & Docs

This repository holds the mirror of the source code of IQM SDK: a collection of libraries for operating IQM's quantum computers.
It also builds and publishes documentation pages for those libraries: [https://docs.meetiqm.com/](https://docs.meetiqm.com/).

This is a "bleeding edge" mirror, i.e. the versions in this mirror are the latest versions of the packages.
Note that public IQM Resonance quantum computers do not always support the latest versions of client packages.
Refer to the Resonance user guides and documentation to find the compatible versions for different quantum computers.

This GitHub repository is a read-only mirror that isn't used for accepting contributions.

**For support, contact `support@meetiqm.com`**.

---

If you need to access an older version of documentation of some package, you can build it locally as follows:

1. Download the source distribution, e.g. `pip download --no-deps --no-binary=:all: iqm-pulse`.
2. Unarchive the downloaded file `tar -xzf iqm_pulse-12.2.0.tar.gz`.
3. `cd` into the directory and build docs with `python -m sphinx docs build/docs`. The rendered docs are now in `build/docs`