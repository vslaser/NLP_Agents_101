from __future__ import annotations
from src.rag.nlp_basics import bow_demo, tokenization_demo, embedding_demo, cosine

def main():
    texts = ["I like apples", "I like bananas", "apples are tasty"]
    vocab, X = bow_demo(texts)
    print("=== Bag of Words ===")
    print("Vocab:", vocab)
    print("Vectors:\n", X)

    print("\n=== Tokenization ===")
    t = tokenization_demo("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Hello there, general Kenobi.")
    print("Tokens:", t["tokens"])

    print("\n=== Sentence Embeddings ===")
    sents = ["I like apples", "I enjoy apples", "The car is fast"]
    vecs = embedding_demo(sents)
    print("sim(apples, enjoy apples) =", cosine(vecs[0], vecs[1]))
    print("sim(apples, car)         =", cosine(vecs[0], vecs[2]))

if __name__ == "__main__":
    main()
