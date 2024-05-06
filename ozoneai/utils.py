import numpy as np

def bit_sim(queries:np.ndarray, docs:np.ndarray):
    """Calculate the bitwise similarity scores for multiple queries against multiple documents
    using numpy broadcasting.

    Args:
        queries (np.ndarray): binary queries matrix [Q x N]
        docs (np.ndarray): binary docs matrix [M x N]

    Returns:
        scores (np.ndarray): similarity scores matrix [Q x M]
    """
    N = docs.shape[1]
    
    if queries.shape[1] != N:
        raise TypeError("query and document dimension doesn't match")
    
    if len(queries.shape) == 1:
        queries = queries[np.newaxis, :]
        
    # Expand queries and docs to 3D for broadcasting
    queries_expanded = queries[:, np.newaxis, :]
    docs_expanded = docs[np.newaxis, :, :]
    
    # Compute bitwise XOR and unpack bits
    dist_pbit = np.bitwise_xor(queries_expanded, docs_expanded)
    dist_bit = np.unpackbits(dist_pbit, axis=2)
    
    # Sum across the last dimension (bit counts)
    dist = np.sum(dist_bit, axis=2)
    
    # Calculate similarity scores and round them
    scores = 1 - dist / (N * 8)
    return np.round(scores, 3)