class EmbeddingModels(object):
    models = [
        "siv-sentence-bitnet-pmbv2-wikid-large",
        "siv-sentence-bitnet-pmbv2-wikid-small"
    ]
    encoders = [
        "paraphrase-multilingual-mpnet-base-v2"
    ]
    encoders_model_map = {
        "paraphrase-multilingual-mpnet-base-v2":[
            "siv-sentence-bitnet-pmbv2-wikid-large",
            "siv-sentence-bitnet-pmbv2-wikid-small"
        ]
    }