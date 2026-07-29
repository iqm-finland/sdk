iqm-data-definitions
====================

A common place for data definitions shared inside IQM. This repository is meant to be independent of any projects other
than necessary 3rd party libraries that are required for code auto-generation. Thus, this repo:

- Contains files that define the data formats. As a concrete example, Protobuf message formats for the quantum computer
  control software stack are located here.

- Contains CI logic to auto-generate and publish serialization and deserialization code as packages that other software
  projects can use.

- Should not contain any hand-written logic that depends on other 3rd party or in-house projects. Note that this is
  subject to change if we decide to include wrapper code for common things in this repo. A separate package should
  then be built from that code with its own dependencies.


Versioning
----------

Breaking changes to data definitions require a major version update.
Major versions are described in the directory paths inside `protos/`. For instance, version 1.x protocol buffers
definitions (.proto files) are located in `protos/iqm/data_definitions/subpackage/v1/*.proto`, where there can be
multiple subpackages for any version.

Backwards-compatible changes can be handled as minor version upgrades. As opposed to being written out in the path name,
minor versions are declared in the package version.
For instance, version 1.2 protocol buffers are still located in the same place as version 1.1 ones, but their
distributable packages have different versions. As an example for Python, the package with version 1.1 protocol buffer
auto-generated code is specified as `iqm-data-definitions==1.1`. That package contains version 1 generated code, where
there is a high-level namespace `iqm` that can contain multiple subpackages with import paths defined as
`iqm.data_definitions.subpackage.v1.*_pb2`.


Workflow
--------

1. Develop .proto files.

2. Alternative ways to test your changes:

    1. Test locally with Docker:
       ``docker run -v path_to_this_repo_root:/home/iqm/idd -w /home/iqm/idd --rm <image_path> tox``.
       Replace <image_path> with the official pipeline image: `gitlab.iqm.fi:5005/iqm/qccsw/iqm-data-definitions:latest`
       , or use other available tag (tags are created by master commit short-sha), or build it by yourself for your own
       platform: ``docker build -t idd:latest -f ci.Dockerfile .``

        - The docker command will mount the current repo root to the image and run ``tox`` there. Note that tox will
          build its environments into `.tox` w.r.t. the Python environment and architecture inside the container.
          Thus, you may need to remove the folder if you desire to run tox outside the container.

        > With the local ``docker run``, running bare ``tox`` invokes the job ``tox -e git_fetch`` which will fail
          because there is no private ssh key setup to access gitlab in the container.
          Make sure you have fetched the latest master. 
       
   2. Test with a local Python environment.
      You need a Python environment with the package ``tox`` installed, see required versions in the `tox.ini` file.
      Run ``tox`` without arguments to: (steps can be run individually as well)

        1. lint protobuf definitions (``tox -e lint``)

        2. fetch the current tip of master from remote (``tox -e git_fetch``).
           Requires this repository to be cloned with ``git`` in order to see previous proto definitions.
        
        3. check protobuf definitions backwards compatibility (``tox -e breaking``)
        
        4. finally, generate wrapper source code files from the .proto files (``tox -e generate``).
           Requires ``protoc`` installed
           (https://developers.google.com/protocol-buffers/docs/reference/python-generated).
           See `ci.Dockerfile` for which version the pipeline is using.

        > Linting and checking compatibility requires ``buf`` to be installed (https://docs.buf.build/installation).
          See `ci.Dockerfile` for which version the pipeline is using.
    
   3. Use only the pipeline.
      Commit and push your changes to the .proto files, and create a merge request.
      Download the artifact from the pipeline job `generate sources merge request`. Extract
      the artifact onto the root of this directory, it should contain the generated sources.

3. Test the new auto-generated code with your desired ways. For Python, editable sources can be installed to an
   environment by ``pip install -e path_to_the_root_of_this_repo``.

4. Repeat from 1. until you're comfortable with your changes and commit the .proto files (not the generated sources).

5. If a new version is needed, create and commit a CHANGELOG entry as well.

6. Create merge request.

7. After the merge request has been merged, a pipeline will kick in automatically to
   1: create a new tag matching the new changelog entry,
   2: auto-generate source code for all defined languages,
   3: finally publish packages from the generates sources. Package version is equal to the tag name.
