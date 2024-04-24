import os, requests, json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import parse_url
from requests.packages.urllib3.util.retry import Retry
from requests.compat import urljoin

from .exception import AuthenticationError

class EmbeddingEndpoints:
    baseurl = "https://speech-kws.ozonetel.com"
    root = urljoin(baseurl, "embeddings/")
    get_embedding = urljoin(root, "text/embedding/get/")
    quantize = urljoin(root, "text/embedding/quantize/")
    url_details = parse_url(baseurl)
    max_retries = 3
    backoff_factor = 0.3

# Embedder Client
class EmbeddingConnector(object):
    """Ozone Embedder Client Application"""
    def __init__(self) -> None:
        super(EmbeddingConnector, self).__init__()
        credfile = os.environ.get('OZAI_API_CREDENTIALS')
        if not credfile:
            raise AuthenticationError(f"No credentials found!\nexport `OZAI_API_CREDENTIALS` before importing the module.")
        if not os.path.exists(credfile):
            raise AuthenticationError(f"Invalid credentials found!\n check if `OZAI_API_CREDENTIALS` valid!")
        
        with open(credfile) as fp:
            credential = json.load(fp)
            if not "username" in credential:
                raise AuthenticationError(f"`username` missing in credentials!")
            if not "bearer_token" in credential:
                raise AuthenticationError(f"`bearer_token` missing in credentials!")
        
        self.username = credential["username"]
        self.bearer_token = credential["bearer_token"]
        self.endpoints = EmbeddingEndpoints
        self.url_details = self.endpoints.url_details
        self.max_retries = self.endpoints.max_retries
        self.backoff_factor = self.endpoints.backoff_factor

    def connect(self):
        # creating persistent connection
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor
        )
        adapter = HTTPAdapter(max_retries=retries)
        scheme = self.url_details.scheme
        self.connection = requests.Session()
        self.connection.mount(scheme, adapter)

    def close(self):
        self.connection.close()
    
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()