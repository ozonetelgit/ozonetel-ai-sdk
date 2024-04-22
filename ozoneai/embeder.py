from .connector import EmbeddingConnector
import numpy as np
from .models import EmbeddingModels

class Embeddings(object):
    def __init__(self, model) -> None:
        self.name = model
        
    def parse(self, response):
        if not "is_error" in response:
            raise AssertionError("invalid input!!")

        if response["is_error"] == 0:
            if "embedding" in response:
                self.embedding = np.array(response["embedding"]).astype("uint8")
                self.bits = np.unpackbits(self.embedding)
            else:
                raise AssertionError("Unable to extract embedding!")
        else:
            raise AssertionError("Unable to extract embedding!")
        return self
        
    def bits(self):
        return self.bits
    
    def ubinary(self):
        return self.embedding
    
    def binary(self):
        return (self.embedding - 128).astype(np.int8)
        

class TextEmbedding(object):
    """Ozone Embedder Client Application"""
    def __init__(self) -> None:
        super(TextEmbedding, self).__init__()
        self.connector = EmbeddingConnector()
        self.models = EmbeddingModels.models

    def connect(self):
        self.connector.connect()

    def close(self):
        self.connector.close()
    
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_embedding(self, text, model):
        """
        text: input text
        model: 
            "siv-sentence-bitnet-pmbv2-wikid-large" or,
            "siv-sentence-bitnet-pmbv2-wikid-small" 
        """
        if not model in self.models:
            raise ValueError("invalid model input!\nselect one from {self.models}")
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.connector.bearer_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {
            "input_text": text,
            "embedder_name": model,
        }

        response = self.connector.connection.post(
            self.connector.endpoints.get_embedding,
            headers=headers, 
            data=data
        )
        embeddings = Embeddings(model)
        return embeddings.parse(response.json())
    