# Copyright 2024 IQM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
from http import HTTPStatus
import io
import locale
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from requests import Response
from requests.adapters import BaseAdapter
from six import BytesIO


class FileAdapter(BaseAdapter):
    def __init__(self, set_content_length=True):
        super(FileAdapter, self).__init__()
        self._set_content_length = set_content_length

    def send(self, request, **kwargs):
        """Sends PreparedRequest object. Returns Response object."""
        # Check that the method makes sense. Only support GET
        if request.method not in ("GET", "HEAD"):
            raise ValueError("Invalid request method %s" % request.method)

        url_parts = urlparse(request.url)

        path = url_parts.netloc + f"{str(Path(url2pathname(url_parts.path)))}.json"
        response = Response()
        try:
            response.status_code = HTTPStatus.OK
            response.raw = io.open(path, "rb")
            response.url = request.url
        except IOError as err:
            if err.errno == errno.ENOENT:
                response.status_code = HTTPStatus.NOT_FOUND
            else:
                response.status_code = HTTPStatus.BAD_REQUEST

            # Wrap the error message in a file-like object
            # The error message will be localized, try to convert the string
            # representation of the exception into a byte stream
            resp_str = str(err).encode(locale.getpreferredencoding(False))
            response.raw = BytesIO(resp_str)

        response.raw.release_conn = response.raw.close

        return response

    def close(self):
        pass
