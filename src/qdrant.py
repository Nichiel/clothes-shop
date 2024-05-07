from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from torch import Tensor
import os

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)


def create_collection(collection_name: str, vector_size: int):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,  # Vector size is defined by used model
            distance=models.Distance.DOT
        ),
    )
    print("Create_collection")


def create_double_embedding_collection(collection_name: str, img_vectors_config: dict, text_vectors_config: dict):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            text_vectors_config["label"]: models.VectorParams(size=text_vectors_config["embedding_size"],
                                                              distance=models.Distance.DOT),
            img_vectors_config["label"]: models.VectorParams(size=img_vectors_config["embedding_size"],
                                                             distance=models.Distance.COSINE),
        },
    )
    print("Create_collection")


def add_embedding_to_double_collection(vectors_image, vectors_text, paths: list, collection_name: str,
                                       img_vectors_config: dict, text_vectors_config: dict):

    qdrant_client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx,
                vector={text_vectors_config["label"]: vet.tolist(), img_vectors_config["label"]: vei.tolist()},
                payload={"path": path.split("/")[-1]}
            )
            for idx, (vet, vei, path) in enumerate(zip(vectors_text, vectors_image, paths))
        ],
    )


def add_embedding_do_collection(vectors, paths: list, collection_name: str):
    qdrant_client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=vec.tolist(), payload={"metadata": path}
            )
            for idx, (vec, path) in enumerate(zip(vectors, paths))
        ],
    )


def query_collection(query: Tensor, collection_name: str, data_type: str, result_num: int):
    result = qdrant_client.search(
        collection_name=collection_name,
        # search_params=models.SearchParams(hnsw_ef=128, exact=False),
        query_vector=(data_type, query[0].tolist()),
        limit=result_num,
    )
    return [im.payload["path"] for im in result]
