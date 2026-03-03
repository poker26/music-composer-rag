"""Qdrant vector store management."""
import logging
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)

logger = logging.getLogger(__name__)


class MusicVectorStore:
    def __init__(self, host="localhost", port=6333,
                 collection="music_fragments", dim=512):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.dim = dim

    def ensure_collection(self):
        """Create collection if it does not exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            for field in ["composer", "era", "genre", "key", "instrument"]:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            for field in ["tempo_bpm", "rms", "note_density"]:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="float",
                )
            logger.info("Created collection '%s' with indexes", self.collection)
        else:
            logger.info("Collection '%s' already exists", self.collection)

    def upsert_fragment(self, embedding, payload):
        """Insert a single music fragment. Returns point ID."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        return point_id

    def upsert_batch(self, items):
        """Batch insert. Each item: {embedding, payload}."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=item["embedding"],
                payload=item["payload"],
            )
            for item in items
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Upserted %d fragments", len(points))
        return len(points)

    def search_similar(self, query_vector, limit=10,
                       composer=None, key=None, tempo_range=None):
        """Search for similar fragments with optional metadata filters."""
        conditions = []
        if composer:
            conditions.append(
                FieldCondition(key="composer", match=MatchValue(value=composer))
            )
        if key:
            conditions.append(
                FieldCondition(key="key", match=MatchValue(value=key))
            )
        if tempo_range:
            conditions.append(
                FieldCondition(
                    key="tempo_bpm",
                    range=Range(gte=tempo_range[0], lte=tempo_range[1]),
                )
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return results.points

    def get_stats(self):
        """Get collection statistics."""
        info = self.client.get_collection(self.collection)
        return {
            "total_points": info.points_count,
            "status": info.status.value,
        }
