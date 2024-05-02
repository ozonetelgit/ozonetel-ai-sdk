# Ozonetel AI
## Overview
The Ozonetel AI project is designed to provide a user-friendly interface for software development using Ozonetel's in-house AI libraries, models, and software solutions. It offers seamless integration with Ozonetel's advanced AI capabilities, allowing developers to harness the power of AI to enhance their applications.

## Features
- Text Embedding (Binary Embeddings): The Ozonetel AI project currently offers text embedding functionality, allowing users to convert text into high-dimensional bit vectors for various natural language processing tasks.

## Getting Started
To get started with the Ozonetel AI project, follow the steps below:

1. Set Credentials:
    Before using the text embedding feature, set your credentials by importing the os module and setting the `OZAI_API_CREDENTIALS` environment variable to point to your credentials file.
    
    Example:
    
    ```python
    import os
    os.environ["OZAI_API_CREDENTIALS"] = "./cred.json"
    ```
3. Text Embedding Extraction
    Text embedding converts textual data into numerical representations, aiding natural language processing tasks. By capturing semantic meaning, it enhances sentiment analysis, document classification, and named entity recognition. Efficient and transferable, embeddings facilitate faster computation and enable machine learning models to better understand and process text. `QuantizeSentenceEmbedding` quantizes base embeddings and represents in bits.

   Example:
    ```python
    # Import `QuantizeSentenceEmbedding` class from the `ozoneai.embeddings` module.
    from ozoneai.embeddings import QuantizeSentenceEmbedding
    
    # Extract Embeddings: Use the `quantize` method to obtain quantised embeddings for given texts .
    # Supported models encoders are `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` and `BAAI/bge-m3`
    # Alternatively if you have stored these models in local directory you can use like `/path/to/paraphrase-multilingual-mpnet-base-v2` or `/path/to/bge-m3`
    with QuantizeSentenceEmbedding(
        endcoder_modelid="BAAI/bge-m3") as embedder:
        emb = embedder.encode(["Try me Out"])
        emb_quantised = embedder.quantize(emb, model="sieve-bge-m3-en-aug-v1") # max limit 20 vectors per request
    
    # Access Embedding Attributes: Retrieve various attributes of the embedding object, such as bits, unsigned binary, and signed binary.
    
    # Get bit representation
    embedding_bits = emb_quantised.bits
    
    # Get unsigned binary
    embedding_ubin = emb_quantised.ubinary()
    
    # Get signed binary
    embedding_bin = emb_quantised.binary()
    ```

    If you want to check available embeddings you can do it as follows
    ```python
    from ozoneai.embeddings import list_models

    list_models()
    ```

## Examples

- [search and index](https://github.com/ozonetelgit/ozonetel-ai-sdk/blob/7d693eb3f012b62ec2bcda6758cc5a4e4d18fae7/examples/search-index/Text%20Indexing%20using%20OzoneAI%20Embeddings%20Faiss.ipynb)

## License
The Ozonetel AI project is licensed under the MIT License. Please refer to the LICENSE file for more information.

