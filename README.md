# Ozonetel AI
## Overview
The Ozonetel AI project is designed to provide a user-friendly interface for software development using Ozonetel's in-house AI libraries, models, and software solutions. It offers seamless integration with Ozonetel's advanced AI capabilities, allowing developers to harness the power of AI to enhance their applications.

## Features
- Text Embedding (Bit Embeddings): The Ozonetel AI project currently offers text embedding functionality, allowing users to convert text into high-dimensional bit vectors for various natural language processing tasks.

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
    # Import `QuantizeSentenceEmbedding` class from the `ozoneai.embeder` module.
    from ozoneai.embeder import QuantizeSentenceEmbedding
    
    # Extract Embeddings: Use the `quantize` method to obtain quantised embeddings for given texts .
    with QuantizeSentenceEmbedding(
        endcoder_modelid="sentence-transformers/paraphrase-multilingual-mpnet-base-v2") as embeder:

        emb = embeder.encode(["Try me Out"])
        emb_quantised = embeder.quantize(emb, model="siv-sentence-bitnet-pmbv2-wikid-large") # max limit 20 vectors per request
    
    # Access Embedding Attributes: Retrieve various attributes of the embedding object, such as bits, unsigned binary, and signed binary.
    
    # Get bit representation
    embedding_bits = emb_quantised.bits
    
    # Get unsigned binary
    embedding_ubin = emb_quantised.ubinary()
    
    # Get signed binary
    embedding_bin = emb_quantised.binary()
    ```



## License
The Ozonetel AI project is licensed under the MIT License. Please refer to the LICENSE file for more information.

