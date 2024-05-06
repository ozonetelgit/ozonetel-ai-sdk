import numpy as np, os, json
from typing import List

from .connector import EmbeddingConnector
from .models import EmbeddingModels
from .exception import LimitError
from sentence_transformers import SentenceTransformer


def list_models():
    for endcoder_modelid, modelids in EmbeddingModels.encoders_model_map.items():
        print(f"""endcoder_modelid: {endcoder_modelid}, model(s):""")
        for modelid in modelids:
            print(f"""\t{modelid}""")
        print("\n")

class Embeddings(object):
    def __init__(self, model) -> None:
        self.name = model
        
    def parse(self, response):
        if not "is_error" in response:
            raise AssertionError("API Error!")

        if response["is_error"] == 0:
            if "embedding" in response:
                self.embedding = np.array(response["embedding"]).astype("uint8")
                self.bits = np.unpackbits(self.embedding, axis=1)
            else:
                raise AssertionError("could not embed!")
        else:
            raise AssertionError("process error!")
        return self
        
    def bits(self):
        return self.bits
    
    def ubinary(self):
        return self.embedding
    
    def binary(self):
        return (self.embedding - 128).astype(np.int8)
    
class BinarizedEmbeddings(object):
    def __init__(self, model) -> None:
        self.name = model
        
    def parse(self, response):
        if not "is_error" in response:
            raise AssertionError("API Error!")

        if response["is_error"] == 0:
            if "embedding" in response:
                self.embedding = np.array(response["embedding"]).astype("uint8")
                self.bits = np.unpackbits(self.embedding, axis=1)
            else:
                raise AssertionError("could not embed!")
        else:
            raise AssertionError("process error!")
        return self
        
    def bits(self):
        return self.bits
    
    def ubinary(self):
        return self.embedding
    
    def binary(self):
        return (self.embedding - 128).astype(np.int8)
        

class TextEmbedding(object):
    """Ozone Embedder Client Application"""
    def __init__(self, credential=None) -> None:
        super(TextEmbedding, self).__init__()
        self.connector = EmbeddingConnector(credential)
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
        
    def __del__(self):
        self.close()

    def get_embedding(self, texts:List[str], model:str):
        f"""
        text: input text
        model: {"or, ".join(self.models)}
        """
        if len(texts) > 20:
            raise LimitError("max limit (20) exceeded!")
        
        if not model in self.models:
            raise ValueError("invalid model input!\nselect one from {self.models}")
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.connector.bearer_token}",
            "Content-Type": "application/json",
        }

        data = {
            "texts": texts,
            "embedder_name": model
        }

        response = self.connector.connection.post(
            self.connector.endpoints.get_embedding,
            headers=headers, 
            json=data
        )
        embeddings = Embeddings(model)
        r = response.json()
        return embeddings.parse(r)
    
    def get_binary_embeddings(self, texts:List[str], model:str):
        f"""
        text: input text
        model: {"or, ".join(self.models)}
        """
        self.connect()
        try:        
            if len(texts) > 20:
                raise LimitError("max limit (20) exceeded!")
            
            if not model in self.models:
                raise ValueError("invalid model input!\nselect one from {self.models}")
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.connector.bearer_token}",
                "Content-Type": "application/json",
            }

            data = {
                "texts": texts,
                "embedder_name": model
            }

            response = self.connector.connection.post(
                self.connector.endpoints.get_embedding,
                headers=headers, 
                json=data
            )
            embeddings = Embeddings(model)
            r = response.json()
            return embeddings.parse(r)
        except Exception as e:
            raise e
        finally:
            self.close()

class BinarizeEmbedding(object):
    """Ozone Embedder Client Application"""
    def __init__(self, endcoder_modelid:str = "sieve-bge-m3-en-aug-v1", credential=None) -> None:
        super(BinarizeEmbedding, self).__init__()
        self.connector = EmbeddingConnector(credential)
        self.models = EmbeddingModels.models
        self.allowed_encoders = EmbeddingModels.encoders
        self.model_maps = EmbeddingModels
        self.encoder_name = os.path.basename(endcoder_modelid.rstrip("/"))
        
        if not self.encoder_name in self.allowed_encoders:
            raise NotImplementedError(f"""Encoder provided is not supported! choose one of {self.allowed_encoders} .""")
        
    def connect(self):
        self.connector.connect()

    def close(self):
        self.connector.close()
    
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def __del__(self):
        self.close()

    def binarize(self, embedding: np.ndarray, model:str):
        f"""
        embedding: text embeddings [ no_of_tokens * embeddings_dim ]
        model: {"or, ".join(self.models)}
        """
        if not model in self.model_maps.encoders_model_map[self.encoder_name]:
            raise ValueError(f"invalid model input!\n {model} is not type of {self.encoder_name}")
        
        if len(embedding.shape) == 1:
            embedding = np.array([embedding])
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.connector.bearer_token}",
            "Content-Type": "application/json",
        }

        data = json.dumps({
            "vectors": embedding.astype(str).tolist(),
            "embedder_name": model,
            "base_model": self.encoder_name
        })

        response = self.connector.connection.post(
            self.connector.endpoints.binarize,
            headers=headers, 
            data=data
        )
        embeddings = BinarizedEmbeddings(model)
        return embeddings.parse(response.json())
    
    def get_binary_embeddings(self, embedding: np.ndarray, model:str):
        f"""
        embedding: text embeddings [ no_of_tokens * embeddings_dim ]
        model: {"or, ".join(self.models)}
        """
        self.connect()
        try:
            if not model in self.model_maps.encoders_model_map[self.encoder_name]:
                raise ValueError(f"invalid model input!\n {model} is not type of {self.encoder_name}")
            
            if len(embedding.shape) == 1:
                embedding = np.array([embedding])
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.connector.bearer_token}",
                "Content-Type": "application/json",
            }

            data = json.dumps({
                "vectors": embedding.astype(str).tolist(),
                "embedder_name": model,
                "base_model": self.encoder_name
            })

            response = self.connector.connection.post(
                self.connector.endpoints.binarize,
                headers=headers, 
                data=data
            )
            embeddings = BinarizedEmbeddings(model)
            return embeddings.parse(response.json())
        except Exception as e:
            raise e
        finally:
            self.close()
            
class BinarizeSentenceEmbedding(object):
    """Ozone Embedder Client Application"""
    def __init__(self, endcoder_modelid:str ="", device:str ="cpu", credential=None) -> None:
        super(BinarizeSentenceEmbedding, self).__init__()
        self.connector = EmbeddingConnector(credential)
        self.models = EmbeddingModels.models
        self.allowed_encoders = EmbeddingModels.encoders
        self.model_maps = EmbeddingModels
        
        self.encoder_name = os.path.basename(endcoder_modelid.rstrip("/"))
        
        if not self.encoder_name in self.allowed_encoders:
            raise NotImplementedError(f"""Encoder provided is not supported! choose one of {self.allowed_encoders} .""")
        
        self.embedder = SentenceTransformer(endcoder_modelid, device=device)

    def connect(self):
        self.connector.connect()

    def close(self):
        self.connector.close()
    
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def __del__(self):
        del self.embedder
        self.close()

    def encode(self, texts:List[str], batch_size: int =4, verbose:bool =False):
        """_summary_

        Args:
            texts (List[str]): _description_
            batch_size (int, optional): _description_. Defaults to 4.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return self.embedder.encode(texts, batch_size=batch_size, show_progress_bar=verbose)
        
    def binarize(self, embedding: np.ndarray, model:str):
        f"""
        embedding: text embeddings [ no_of_tokens * embeddings_dim ]
        model: {"or, ".join(self.models)}
        """
        if not model in self.model_maps.encoders_model_map[self.encoder_name]:
            raise ValueError(f"invalid model input!\n {model} is not type of {self.encoder_name}")
        
        if len(embedding.shape) == 1:
            embedding = np.array([embedding])
            
        if embedding.shape[0] > 20:
            raise LimitError("max limit (20) exceeded!")
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.connector.bearer_token}",
            "Content-Type": "application/json",
        }

        data = json.dumps({
            "vectors": embedding.astype(str).tolist(),
            "embedder_name": model,
            "base_model": self.encoder_name
        })

        response = self.connector.connection.post(
            self.connector.endpoints.binarize,
            headers=headers, 
            data=data
        )

        embeddings = BinarizedEmbeddings(model)
        r = response.json()
        return embeddings.parse(r)

    def get_binary_embeddings(self, embedding: np.ndarray, model:str):
        f"""
        embedding: text embeddings [ no_of_tokens * embeddings_dim ]
        model: {"or, ".join(self.models)}
        """
        self.connect()
        try:
            if not model in self.model_maps.encoders_model_map[self.encoder_name]:
                raise ValueError(f"invalid model input!\n {model} is not type of {self.encoder_name}")
            
            if len(embedding.shape) == 1:
                embedding = np.array([embedding])
                
            if embedding.shape[0] > 20:
                raise LimitError("max limit (20) exceeded!")
            
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.connector.bearer_token}",
                "Content-Type": "application/json",
            }

            data = json.dumps({
                "vectors": embedding.astype(str).tolist(),
                "embedder_name": model,
                "base_model": self.encoder_name
            })

            response = self.connector.connection.post(
                self.connector.endpoints.binarize,
                headers=headers, 
                data=data
            )

            embeddings = BinarizedEmbeddings(model)
            r = response.json()
            return embeddings.parse(r)
        except Exception as e:
            raise e
        finally:
            self.close()
        
