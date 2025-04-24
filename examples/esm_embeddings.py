import argparse

from rcsb_embedding_model import RcsbStructureEmbedding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--file_format', type=str)
    parser.add_argument('--chain', type=str)
    args = parser.parse_args()

    model = RcsbStructureEmbedding()
    res_embedding = model.residue_embedding(
        src_structure=args.file,
        src_format=args.file_format,
        chain_id=args.chain
    )
    structure_embedding = model.aggregator_embedding(
        res_embedding
    )

    print(res_embedding.shape, structure_embedding.shape)
