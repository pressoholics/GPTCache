from typing import List, Optional, Union
import numpy as np
from opensearchpy import OpenSearch, exceptions

from gptcache.manager.vector_data.base import VectorBase, VectorData


class AWSOpenSearchVectorStore(VectorBase):
    def __init__(
        self,
        host: str,
        port: int,
        use_ssl: bool,
        verify_certs: bool,
        http_auth: tuple,
        index_name: str,
        dimension: int,
        top_k: int = 1
    ):
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            http_auth=http_auth
        )
        self.index_name = index_name
        self.dimension = dimension
        self.top_k = top_k

        # Check if the index exists, if not create one
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        # Settings for the OpenSearch index, like number of shards and replicas
                    },
                    "mappings": {
                        "properties": {
                            "vector": {
                                "type": "dense_vector",
                                "dims": self.dimension
                            }
                            # Additional properties can be added here
                        }
                    }
                }
            )

    def mul_add(self, datas: List[VectorData]):
        bulk_data = [
            {
                "_index": self.index_name,
                "_id": str(data.id),
                "_source": {
                    "vector": data.data.tolist()  # Convert np.ndarray to list
                }
            }
            for data in datas
        ]
        OpenSearch.helpers.bulk(self.client, bulk_data)

    def search(self, data: np.ndarray, top_k: int = -1):
        if top_k == -1:
            top_k = self.top_k

        query = {
            "size": top_k,
            "query": {
                "knn": {
                    "field": "vector",
                    "vector": data.tolist(),
                    "k": top_k
                }
            }
        }
        response = self.client.search(body=query, index=self.index_name)
        # Format of response needs to be adjusted based on the actual output
        return [(hit['_score'], hit['_id']) for hit in response['hits']['hits']]

    def rebuild(self, ids=None) -> bool:
        # In OpenSearch, index rebuilding is not typically needed as it's handled automatically.
        return True

    def delete(self, ids) -> bool:
        try:
            for data_id in ids:
                self.client.delete(index=self.index_name, id=str(data_id))
            return True
        except exceptions.NotFoundError:
            return False

    def get_embeddings(self, data_id: Union[int, str]) -> Optional[np.ndarray]:
        try:
            result = self.client.get(index=self.index_name, id=str(data_id))
            return np.array(result['_source']['vector'])
        except exceptions.NotFoundError:
            return None

    def update_embeddings(self, data_id: Union[int, str], emb: np.ndarray):
        document = {
            "doc": {
                "vector": emb.tolist()
            }
        }
        self.client.update(index=self.index_name, id=str(data_id), body=document)
