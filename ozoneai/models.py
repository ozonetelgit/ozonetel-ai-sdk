class EmbeddingModels(object):
    models = [
        "siv-sentence-bitnet-pmbv2-wikid-large",
        "siv-sentence-bitnet-pmbv2-wikid-small",
        "sieve-bge-m3-en-aug-v0",
        "sieve-bge-m3-en-aug-v1"
    ]
    encoders = [
        "paraphrase-multilingual-mpnet-base-v2",
        "bge-m3"
    ]
    encoders_model_map = {
        "paraphrase-multilingual-mpnet-base-v2":[
            "siv-sentence-bitnet-pmbv2-wikid-large",
            "siv-sentence-bitnet-pmbv2-wikid-small"
        ],
        "bge-m3":[
            "sieve-bge-m3-en-aug-v0",
            "sieve-bge-m3-en-aug-v1"
        ]
    }