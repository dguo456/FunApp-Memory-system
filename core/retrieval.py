from typing import List, Optional, Dict, Tuple
import asyncpg
import numpy as np
from dataclasses import dataclass
import logging

from core.schemas import (
    MemoryQuery, Memory, MemorySearchResult, RetrievalResponse,
    MemoryType, Character
)
from app.deps import DatabaseManager, EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Encapsulates search filters and parameters"""
    memory_types: Optional[List[MemoryType]] = None
    character_ids: Optional[List[int]] = None
    chapter_range: Optional[Tuple[int, int]] = None
    context_tags: Optional[List[str]] = None
    importance_threshold: float = 0.0


class HybridMemoryRetriever:
    """Hybrid retrieval system combining semantic and keyword search"""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service

    async def search_memories(self, query: MemoryQuery) -> RetrievalResponse:
        """Perform hybrid search combining semantic and keyword approaches"""
        try:
            # Generate query embedding for semantic search
            query_embedding = self.embedding_service.encode_text(query.query_text)

            # Build search filters
            search_filter = SearchFilter(
                memory_types=query.memory_types,
                character_ids=query.character_ids,
                chapter_range=query.chapter_range,
                context_tags=query.context_tags
            )

            # Perform semantic search
            semantic_results = await self._semantic_search(
                query_embedding, search_filter, query.limit, query.similarity_threshold
            )

            # Perform keyword search
            keyword_results = await self._keyword_search(
                query.query_text, search_filter, query.limit
            )

            # Combine and re-rank results
            combined_results = await self._hybrid_rerank(
                semantic_results, keyword_results, query.hybrid_alpha, query.limit
            )

            # Update access counts for retrieved memories
            await self._update_access_counts([result.memory.id for result in combined_results])

            return RetrievalResponse(
                query=query.query_text,
                results=combined_results,
                total_found=len(combined_results),
                query_embedding=query_embedding.tolist()
            )

        except Exception as e:
            logger.error(f"Memory search failed for query '{query.query_text}': {e}")
            raise

    async def _semantic_search(
        self,
        query_embedding: np.ndarray,
        search_filter: SearchFilter,
        limit: int,
        similarity_threshold: float
    ) -> List[MemorySearchResult]:
        """Perform semantic similarity search using vector embeddings"""

        # Build the base query with filters
        base_query, params = self._build_base_query_with_filters(search_filter)

        # Add vector similarity search
        vector_query = f"""
        SELECT m.*, (1 - (m.embedding <=> $1)) as similarity_score
        FROM ({base_query}) m
        WHERE m.embedding IS NOT NULL
          AND (1 - (m.embedding <=> $1)) >= $2
        ORDER BY similarity_score DESC
        LIMIT $3
        """

        # Parameters: embedding (as string), threshold, limit, plus filter params
        vector_params = [str(query_embedding.tolist()), similarity_threshold, limit] + params

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(vector_query, *vector_params)

        results = []
        for i, row in enumerate(rows):
            memory = self._row_to_memory(row)
            result = MemorySearchResult(
                memory=memory,
                similarity_score=float(row['similarity_score']),
                rank=i + 1,
                retrieval_method="semantic"
            )
            results.append(result)

        logger.info(f"Semantic search returned {len(results)} results")
        return results

    async def _keyword_search(
        self,
        query_text: str,
        search_filter: SearchFilter,
        limit: int
    ) -> List[MemorySearchResult]:
        """Perform keyword-based full-text search"""

        # Build the base query with filters
        base_query, params = self._build_base_query_with_filters(search_filter)

        # Add full-text search
        fts_query = f"""
        SELECT m.*, ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as fts_score
        FROM ({base_query}) m
        WHERE to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)
        ORDER BY fts_score DESC
        LIMIT $2
        """

        # Parameters: query_text, limit, plus filter params
        fts_params = [query_text, limit] + params

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(fts_query, *fts_params)

        results = []
        for i, row in enumerate(rows):
            memory = self._row_to_memory(row)
            result = MemorySearchResult(
                memory=memory,
                similarity_score=float(row['fts_score']),
                rank=i + 1,
                retrieval_method="keyword"
            )
            results.append(result)

        logger.info(f"Keyword search returned {len(results)} results")
        return results

    def _build_base_query_with_filters(self, search_filter: SearchFilter) -> Tuple[str, List]:
        """Build base query with all filters applied"""
        query_parts = ["SELECT * FROM memories WHERE 1=1"]
        params = []
        param_count = 0

        # Memory type filter
        if search_filter.memory_types:
            param_count += 1
            type_values = [mt.value for mt in search_filter.memory_types]
            query_parts.append(f"AND memory_type = ANY(${param_count})")
            params.append(type_values)

        # Character filter
        if search_filter.character_ids:
            param_count += 1
            query_parts.append(f"AND character_id = ANY(${param_count})")
            params.append(search_filter.character_ids)

        # Chapter range filter
        if search_filter.chapter_range:
            start_chapter, end_chapter = search_filter.chapter_range
            param_count += 1
            query_parts.append(f"AND chapter_number >= ${param_count}")
            params.append(start_chapter)
            param_count += 1
            query_parts.append(f"AND chapter_number <= ${param_count}")
            params.append(end_chapter)

        # Context tags filter
        if search_filter.context_tags:
            param_count += 1
            query_parts.append(f"AND context_tags && ${param_count}")
            params.append(search_filter.context_tags)

        # Importance threshold
        if search_filter.importance_threshold > 0:
            param_count += 1
            query_parts.append(f"AND importance_score >= ${param_count}")
            params.append(search_filter.importance_threshold)

        return " ".join(query_parts), params

    async def _hybrid_rerank(
        self,
        semantic_results: List[MemorySearchResult],
        keyword_results: List[MemorySearchResult],
        alpha: float,
        limit: int
    ) -> List[MemorySearchResult]:
        """Combine and re-rank results from semantic and keyword search"""

        # Create a map of memory_id to results
        all_results = {}

        # Add semantic results
        for result in semantic_results:
            memory_id = result.memory.id
            all_results[memory_id] = {
                'memory': result.memory,
                'semantic_score': result.similarity_score,
                'keyword_score': 0.0,
                'methods': ['semantic']
            }

        # Add keyword results
        for result in keyword_results:
            memory_id = result.memory.id
            if memory_id in all_results:
                all_results[memory_id]['keyword_score'] = result.similarity_score
                all_results[memory_id]['methods'].append('keyword')
            else:
                all_results[memory_id] = {
                    'memory': result.memory,
                    'semantic_score': 0.0,
                    'keyword_score': result.similarity_score,
                    'methods': ['keyword']
                }

        # Normalize scores and calculate hybrid score
        max_semantic = max([r['semantic_score'] for r in all_results.values()]) if all_results else 1.0
        max_keyword = max([r['keyword_score'] for r in all_results.values()]) if all_results else 1.0

        hybrid_results = []
        for memory_id, data in all_results.items():
            normalized_semantic = data['semantic_score'] / max_semantic if max_semantic > 0 else 0
            normalized_keyword = data['keyword_score'] / max_keyword if max_keyword > 0 else 0

            # Hybrid score: alpha * semantic + (1-alpha) * keyword
            hybrid_score = alpha * normalized_semantic + (1 - alpha) * normalized_keyword

            method = 'hybrid' if len(data['methods']) > 1 else data['methods'][0]

            result = MemorySearchResult(
                memory=data['memory'],
                similarity_score=hybrid_score,
                rank=0,  # Will be set after sorting
                retrieval_method=method
            )
            hybrid_results.append(result)

        # Sort by hybrid score and assign ranks
        hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
        for i, result in enumerate(hybrid_results[:limit]):
            result.rank = i + 1

        logger.info(f"Hybrid reranking produced {len(hybrid_results[:limit])} final results")
        return hybrid_results[:limit]

    async def _update_access_counts(self, memory_ids: List[int]):
        """Update access counts for retrieved memories"""
        if not memory_ids:
            return

        try:
            async with self.db_manager.get_connection() as conn:
                query = """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE id = ANY($1)
                """
                await conn.execute(query, memory_ids)

        except Exception as e:
            logger.error(f"Failed to update access counts: {e}")

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            id=row['id'],
            memory_type=MemoryType(row['memory_type']),
            character_id=row['character_id'],
            related_entity_id=row['related_entity_id'],
            content=row['content'],
            summary=row['summary'],
            chapter_number=row['chapter_number'],
            context_tags=list(row['context_tags']) if row['context_tags'] else [],
            embedding=eval(row['embedding']) if row['embedding'] else None,
            importance_score=float(row['importance_score']),
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )

    async def get_memory_by_id(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a specific memory by ID"""
        try:
            async with self.db_manager.get_connection() as conn:
                row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)

                if row:
                    await conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = $1",
                        memory_id
                    )
                    return self._row_to_memory(row)

                return None

        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    async def get_character_memories(
        self,
        character_id: int,
        memory_type: Optional[MemoryType] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Get all memories for a specific character"""
        try:
            query = "SELECT * FROM memories WHERE character_id = $1"
            params = [character_id]

            if memory_type:
                query += " AND memory_type = $2"
                params.append(memory_type.value)

            query += " ORDER BY importance_score DESC, created_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_memory(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get character memories for character {character_id}: {e}")
            return []

    async def get_related_memories(
        self,
        memory_id: int,
        limit: int = 10
    ) -> List[MemorySearchResult]:
        """Find memories related to a given memory through relationships or similarity"""
        try:
            # Get the source memory
            source_memory = await self.get_memory_by_id(memory_id)
            if not source_memory or not source_memory.embedding:
                return []

            # Find similar memories based on embeddings
            query_embedding = np.array(source_memory.embedding)

            query = """
            SELECT *, (1 - (embedding <=> $1)) as similarity_score
            FROM memories
            WHERE id != $2
              AND embedding IS NOT NULL
              AND (1 - (embedding <=> $1)) >= 0.6
            ORDER BY similarity_score DESC
            LIMIT $3
            """

            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(query, str(query_embedding.tolist()), memory_id, limit)

            results = []
            for i, row in enumerate(rows):
                memory = self._row_to_memory(row)
                result = MemorySearchResult(
                    memory=memory,
                    similarity_score=float(row['similarity_score']),
                    rank=i + 1,
                    retrieval_method="related"
                )
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Failed to get related memories for {memory_id}: {e}")
            return []

    async def search_by_context(
        self,
        character_id: int,
        context_tags: List[str],
        chapter_range: Optional[Tuple[int, int]] = None,
        limit: int = 20
    ) -> List[Memory]:
        """Search memories by context tags and character"""
        try:
            query = """
            SELECT * FROM memories
            WHERE character_id = $1 AND context_tags && $2
            """
            params = [character_id, context_tags]

            if chapter_range:
                query += " AND chapter_number >= $3 AND chapter_number <= $4"
                params.extend([chapter_range[0], chapter_range[1]])

            query += " ORDER BY importance_score DESC, created_at DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_memory(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to search by context: {e}")
            return []