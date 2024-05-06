# Ozonetel AI
## Overview
The Ozonetel AI project is designed to provide a user-friendly interface for software development using Ozonetel's in-house AI libraries, models, and software solutions. It offers seamless integration with Ozonetel's advanced AI capabilities, allowing developers to harness the power of AI to enhance their applications.

## Features
- Text Embedding (Binary Embeddings): The Ozonetel AI project currently offers text embedding functionality, allowing users to convert text into high-dimensional bit vectors for various natural language processing tasks.

## Getting Started
To get started with the Ozonetel AI project, follow the steps below:

1. Usage
    1. Installation
    
    ```bash
    pip install ozonetel-ai
    ```

    2. compute binary embedding 

    ```python
    from ozoneai.embeddings import BinarizeEmbedding
    from ozoneai.utils import bit_sim

    from sentence_transformers import SentenceTransformer

    credential = {"username":"", "bearer_token":""}

    model = SentenceTransformer("BAAI/bge-m3")
    binary_encoder = BinarizeEmbedding(endcoder_modelid="BAAI/bge-m3",credential=credential)
    ```
1. Set Credentials:
    Before using the text embedding feature, set your credentials by importing the os module and setting the `OZAI_API_CREDENTIALS` environment variable to point to your credentials file.
    
    Example:
    
    ```python
    import os
    os.environ["OZAI_API_CREDENTIALS"] = "./cred.json"
    ```
    or,
    ``` python
    credential = {"username":"", "bearer_token":""}
    ```
2. Text Embedding Extraction
    Text embedding converts textual data into numerical representations, aiding natural language processing tasks. By capturing semantic meaning, it enhances sentiment analysis, document classification, and named entity recognition. Efficient and transferable, embeddings facilitate faster computation and enable machine learning models to better understand and process text. `BinarizeSentenceEmbedding` binarizes base embeddings and represents in bits.

   Example:
   - Binary Embedding extraction
        ```python
        # Import `BinarizeSentenceEmbedding` class from the `ozoneai.embeddings` module.
        from ozoneai.embeddings import BinarizeSentenceEmbedding
        
        # Extract Embeddings: Use the `binarize` method to obtain binarized embeddings for given texts .
        # Supported models encoders are `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` and `BAAI/bge-m3`
        # Alternatively if you have stored these models in local directory you can use like `/path/to/paraphrase-multilingual-mpnet-base-v2` or `/path/to/bge-m3`
        
        # Note: for safest use `OZAI_API_CREDENTIALS`. In that case `credential` argument can be ignored. if `credential` is not defined `OZAI_API_CREDENTIALS` will be considered.
        binary_encoder = BinarizeSentenceEmbedding(endcoder_modelid="BAAI/bge-m3", credential = credential)

        emb = binary_encoder.encode(["Try me Out"])
        emb_binarized = binary_encoder.get_binary_embeddings(emb, model="sieve-bge-m3-en-aug-v1") # max limit 20 vectors per request
        
        # Access Embedding Attributes: Retrieve various attributes of the embedding object, such as bits, unsigned binary, and signed binary.
        # Get binary representation
        embeddings = emb_binarized.embedding

        ```

    - List the supported models
        
        ```python
        from ozoneai.embeddings import list_models

        list_models()
        ```

    - Document Similarity

        ```python
        from ozoneai.embeddings import BinarizeEmbedding
        from ozoneai.utils import bit_sim

        from sentence_transformers import SentenceTransformer

        credential = {"username":"", "bearer_token":""}

        model = SentenceTransformer("BAAI/bge-m3")
        binary_encoder = BinarizeEmbedding(endcoder_modelid="BAAI/bge-m3",credential=credential)

        # Compute embedding for both lists
        query = ["What is BGE M3?", "Define BM25"]
        document = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
                    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

        query_embeddings = binary_encoder.get_binary_embeddings(model.encode(query), model="sieve-bge-m3-en-aug-v1").embedding
        doc_embeddings = binary_encoder.get_binary_embeddings(model.encode(document), model="sieve-bge-m3-en-aug-v1").embedding

        # Compute similarities
        scores = bit_sim(query_embeddings, doc_embeddings)

        for i, qs in enumerate(scores):
            for j, ds in enumerate(qs):
                print(f"query {i} vs doc {j} : similarity - {round(float(ds), 3)}")
        
        ```
        ```
        output:

            query 0 vs doc 0 : similarity - 0.627
            query 0 vs doc 1 : similarity - 0.537
            query 1 vs doc 0 : similarity - 0.528
            query 1 vs doc 1 : similarity - 0.634
        ```


## Sample Applications

### Search And Index:

```python
import numpy as np, os
from ozoneai.embeddings import BinarizeEmbedding

from sentence_transformers import SentenceTransformer

# define credentials
# os.environ["OZAI_API_CREDENTIALS"] = "./cred.json"
credential = {"username":"", "bearer_token":""}

base_model = SentenceTransformer("BAAI/bge-m3")
binary_encoder = BinarizeEmbedding(endcoder_modelid="BAAI/bge-m3",credential=credential)

# Documents to be Indexed
docs = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
    "Artifical intelligence is all about enabling computers to simulate human capabilities.",
    "Artifical intelligence is there to help human",
    "Language moodel has lot of impact on artifical intelligence",
    "Artifical intelligence is reigning in industry",
    "ROI is one the useful measure for any investors",
    "Understand before you invest"
]

# Extract Embeddings for documents to be indexed
docs_emb = base_model.encode(docs)

# Binarise embeedings
docs_emb_binarized = binary_encoder.get_binary_embeddings(docs_emb, model="sieve-bge-m3-en-aug-v1")

# Searching

# query_text = "the girl was sitting in the park"
query_text = "I love Artifical Intellegence"
# query_text = "Artifical intellegence helps optimising software"
# query_text = "what is machine learning"

# Extract Embeddings query

# Compute sentence embedding using SentenceTransformers (i.e. bge-m3 or paraphrase-multilingual-mpnet-base-v2)
query_emb = base_model.encode([query_text])

# Binarise embeedings
query_emb_binarized = binary_encoder.get_binary_embeddings(query_emb, model="sieve-bge-m3-en-aug-v1")

scores = bit_sim(query_emb_binarized.embedding, docs_emb_binarized.embedding).flatten()

topn = 5
topn_indices = np.argsort(-scores)[:topn]

# Print search result in order
for i, doci in enumerate(topn_indices):
    print(f"{i}. {docs[doci]} [{doci}] ({scores[doci]})")
```
    
[try more..](https://github.com/ozonetelgit/ozonetel-ai-sdk/blob/main/examples/search-index/)

## Benchmarks
### Classification
| S.N | Dataset                                   | paraphrase-multilingual-mpnet-base-v2 | siv-sentence-bitnet-pmbv2-wikid-small | bge-m3 | sieve-bge-m3-en-aug-v0 | sieve-bge-m3-en-aug-v1 |
| --- | ----------------------------------------- | ------------------------------------- | ------------------------------------- | ------ | ---------------------- | ---------------------- |
| 1   | Amazon Counterfactual Classification(en)  | 75.81                                 | 79.06                                 | 75.63  | 79.23                  | 78.28                  |
| 2   | Amazon Polarity Classification            | 76.41                                 | 70.19                                 | 91.01  | 86.81                  | 85.69                  |
| 3   | Amazon Reviews Classification             | 38.51                                 | 34.29                                 | 46.99  | 43.51                  | 43                     |
| 4   | Banking 77 Classification                 | 81.07                                 | 75.89                                 | 81.93  | 82.06                  | 82.75                  |
| 5   | Emotion Classification                    | 45.83                                 | 40.26                                 | 50.16  | 42.34                  | 42.4                   |
| 6   | Imdb Classification                       | 64.57                                 | 61.14                                 | 87.84  | 85.06                  | 84.44                  |
| 7   | Massive Intent Classification (en)        | 69.32                                 | 65.6                                  | 71.08  | 68.9                   | 70.22                  |
| 8   | Massive Scenario Classification (en)      | 75.35                                 | 70.37                                 | 76.64  | 71.29                  | 72.68                  |
| 9   | MTOP Domain Classification (en)           | 89.24                                 | 87.22                                 | 93.36  | 87.56                  | 88.37                  |
| 10  | MTOP Intent Classification (en)           | 68.69                                 | 69.45                                 | 66.58  | 74.11                  | 74.02                  |
| 11  | Toxic Conversations Classification        | 71.02                                 | 70.26                                 | 72.6   | 68                     | 68.59                  |
| 12  | Tweet Sentiment Extraction Classification | 59.03                                 | 54.49                                 | 63.71  | 56.96                  | 56.91                  |

### STS
| S.N | Dataset       | paraphrase-multilingual-mpnet-base-v2 | siv-sentence-bitnet-pmbv2-wikid-small | bge-m3 | sieve-bge-m3-en-aug-v0 | sieve-bge-m3-en-aug-v1 |
| --- | ------------- | ------------------------------------- | ------------------------------------- | ------ | ---------------------- | ---------------------- |
| 1   | BIOSSES       | 76.27                                 | 65.29                                 | 83.38  | 82.91                  | 83.97                  |
| 2   | SICK-R        | 79.62                                 | 76.01                                 | 79.91  | 75.26                  | 76.5                   |
| 3   | STS12         | 77.9                                  | 71.25                                 | 78.73  | 66.95                  | 68.88                  |
| 4   | STS13         | 85.11                                 | 78.4                                  | 79.6   | 64                     | 69.09                  |
| 5   | STS14         | 80.81                                 | 74.23                                 | 79     | 62.83                  | 67.74                  |
| 6   | STS15         | 87.48                                 | 81.41                                 | 87.81  | 80                     | 82.08                  |
| 7   | STS16         | 83.2                                  | 79.13                                 | 85.4   | 77.87                  | 79.31                  |
| 8   | STS17(en-en)  | 86.99                                 | 85.4                                  | 87.13  | 84.1                   | 85.32                  |
| 9   | STS Benchmark | 86.82                                 | 81.34                                 | 84.85  | 74.13                  | 77.19                  |


## License
The Ozonetel AI project is licensed under the MIT License. Please refer to the LICENSE file for more information.

