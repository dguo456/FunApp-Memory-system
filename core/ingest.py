import json
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from core.schemas import (
    ChapterData, MemoryCreate, Character, User, WorldStateCreate,
    ConsistencyLedgerCreate, ChangeType, ChapterProcessingResult,
    BulkMemoryCreate, BulkMemoryResponse
)
from core.extractors import MemoryExtractor
from app.deps import get_db_connection, get_embedding_service, DatabaseManager
import asyncpg

logger = logging.getLogger(__name__)


class MemoryIngestionService:
    """Service for ingesting and processing memories from chapter data"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embedding_service = get_embedding_service()
        self.extractor = MemoryExtractor()

    async def load_json_data(self, file_path: str) -> List[ChapterData]:
        """Load chapter data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chapters = []
            for item in data:
                chapter = ChapterData(
                    chapter_number=item['chapter_number'],
                    synopsis=item['synopsis']
                )
                chapters.append(chapter)

            logger.info(f"Loaded {len(chapters)} chapters from {file_path}")
            return chapters

        except Exception as e:
            logger.error(f"Failed to load JSON data from {file_path}: {e}")
            raise

    async def process_all_chapters(
        self,
        chapters: List[ChapterData],
        batch_size: int = 10
    ) -> List[ChapterProcessingResult]:
        """Process all chapters in batches"""
        results = []

        # Load characters and user
        characters = await self._load_characters()
        user = await self._load_user()

        # Process chapters in batches
        for i in range(0, len(chapters), batch_size):
            batch = chapters[i:i + batch_size]
            logger.info(f"Processing chapter batch {i//batch_size + 1}: chapters {batch[0].chapter_number}-{batch[-1].chapter_number}")

            batch_results = await asyncio.gather(*[
                self.process_chapter(chapter, characters, user)
                for chapter in batch
            ])
            results.extend(batch_results)

        return results

    async def process_chapter(
        self,
        chapter_data: ChapterData,
        characters: Dict[str, Character],
        user: User
    ) -> ChapterProcessingResult:
        """Process a single chapter and extract memories"""
        try:
            logger.info(f"Processing Chapter {chapter_data.chapter_number}")

            # Extract memories using rule-based extraction
            extracted_memories = self.extractor.extract_memories_from_chapter(
                chapter_data, characters, user
            )

            # Generate embeddings for memories
            embeddings = await self._add_embeddings_to_memories(extracted_memories)

            # Store memories in database
            created_count, updated_count = await self._store_memories(extracted_memories, embeddings)

            # Update world state
            world_state_updated = await self._update_world_state(chapter_data)

            # Check for consistency issues
            consistency_issues = await self._check_consistency(extracted_memories)

            result = ChapterProcessingResult(
                chapter_number=chapter_data.chapter_number,
                memories_created=created_count,
                memories_updated=updated_count,
                world_state_updated=world_state_updated,
                consistency_issues=consistency_issues
            )

            logger.info(f"Chapter {chapter_data.chapter_number} processed: {created_count} created, {updated_count} updated")
            return result

        except Exception as e:
            logger.error(f"Failed to process Chapter {chapter_data.chapter_number}: {e}")
            raise

    async def _load_characters(self) -> Dict[str, Character]:
        """Load all characters from database"""
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch("SELECT id, name, description, created_at, updated_at FROM characters")

            characters = {}
            for row in rows:
                char = Character(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                characters[char.name] = char

            logger.info(f"Loaded {len(characters)} characters")
            return characters

    async def _load_user(self) -> User:
        """Load user from database"""
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow("SELECT id, name, created_at FROM users LIMIT 1")

            if not row:
                # Create default user if none exists
                user_id = await conn.fetchval(
                    "INSERT INTO users (name) VALUES ($1) RETURNING id",
                    "User"
                )
                row = await conn.fetchrow("SELECT id, name, created_at FROM users WHERE id = $1", user_id)

            return User(
                id=row['id'],
                name=row['name'],
                created_at=row['created_at']
            )

    async def _add_embeddings_to_memories(self, memories: List[MemoryCreate]) -> List[List[float]]:
        """Generate embeddings for memories and return them"""
        if not memories:
            return []

        # Extract texts for batch embedding
        texts = [memory.content for memory in memories]

        # Generate embeddings in batch
        embeddings = self.embedding_service.encode_batch(texts)

        # Convert to list format
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        logger.info(f"Generated embeddings for {len(memories)} memories")
        return embeddings_list

    async def _store_memories(self, memories: List[MemoryCreate], embeddings: List[List[float]]) -> Tuple[int, int]:
        """Store memories in database"""
        if not memories:
            return 0, 0

        created_count = 0
        updated_count = 0

        async with self.db_manager.get_connection() as conn:
            async with conn.transaction():
                for i, memory in enumerate(memories):
                    embedding = embeddings[i] if i < len(embeddings) else None

                    # Check if similar memory already exists
                    existing_id = await self._find_similar_memory(conn, memory)

                    if existing_id:
                        # Update existing memory
                        await self._update_existing_memory(conn, existing_id, memory, embedding)
                        updated_count += 1
                    else:
                        # Create new memory
                        await self._create_new_memory(conn, memory, embedding)
                        created_count += 1

        return created_count, updated_count

    async def _find_similar_memory(self, conn: asyncpg.Connection, memory: MemoryCreate) -> Optional[int]:
        """Find existing similar memory"""
        query = """
        SELECT id FROM memories
        WHERE memory_type = $1
          AND character_id = $2
          AND related_entity_id = $3
          AND chapter_number = $4
        LIMIT 1
        """

        row = await conn.fetchrow(
            query,
            memory.memory_type.value,
            memory.character_id,
            memory.related_entity_id,
            memory.chapter_number
        )

        return row['id'] if row else None

    async def _create_new_memory(self, conn: asyncpg.Connection, memory: MemoryCreate, embedding: Optional[List[float]] = None) -> int:
        """Create new memory in database"""
        query = """
        INSERT INTO memories (
            memory_type, character_id, related_entity_id, content, summary,
            chapter_number, context_tags, embedding, importance_score
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        """

        # Convert embedding list to string format for pgvector
        embedding_str = str(embedding) if embedding else None

        memory_id = await conn.fetchval(
            query,
            memory.memory_type.value,
            memory.character_id,
            memory.related_entity_id,
            memory.content,
            memory.summary,
            memory.chapter_number,
            memory.context_tags,
            embedding_str,
            memory.importance_score
        )

        # Log the creation in consistency ledger
        await self._log_consistency_change(
            conn, memory_id, ChangeType.CREATED,
            None, memory.content, f"Memory created from Chapter {memory.chapter_number}"
        )

        return memory_id

    async def _update_existing_memory(self, conn: asyncpg.Connection, memory_id: int, new_memory: MemoryCreate, embedding: Optional[List[float]] = None):
        """Update existing memory"""
        # Get old content for logging
        old_content = await conn.fetchval("SELECT content FROM memories WHERE id = $1", memory_id)

        # Update memory
        query = """
        UPDATE memories
        SET content = $1, summary = $2, context_tags = $3, embedding = $4,
            importance_score = $5, updated_at = CURRENT_TIMESTAMP
        WHERE id = $6
        """

        # Convert embedding list to string format for pgvector
        embedding_str = str(embedding) if embedding else None

        await conn.execute(
            query,
            new_memory.content,
            new_memory.summary,
            new_memory.context_tags,
            embedding_str,
            new_memory.importance_score,
            memory_id
        )

        # Log the update in consistency ledger
        await self._log_consistency_change(
            conn, memory_id, ChangeType.UPDATED,
            old_content, new_memory.content,
            f"Memory updated from Chapter {new_memory.chapter_number}"
        )

    async def _log_consistency_change(
        self,
        conn: asyncpg.Connection,
        memory_id: int,
        change_type: ChangeType,
        old_content: Optional[str],
        new_content: str,
        reason: str
    ):
        """Log change in consistency ledger"""
        query = """
        INSERT INTO consistency_ledger (
            memory_id, change_type, old_content, new_content, change_reason
        ) VALUES ($1, $2, $3, $4, $5)
        """

        await conn.execute(
            query,
            memory_id,
            change_type.value,
            old_content,
            new_content,
            reason
        )

    async def _update_world_state(self, chapter_data: ChapterData) -> bool:
        """Update world state for the chapter"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Check if world state already exists for this chapter
                existing = await conn.fetchrow(
                    "SELECT id FROM world_states WHERE chapter_number = $1",
                    chapter_data.chapter_number
                )

                if not existing:
                    # Create new world state entry
                    await conn.execute(
                        "INSERT INTO world_states (chapter_number, state_description) VALUES ($1, $2)",
                        chapter_data.chapter_number,
                        chapter_data.synopsis
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to update world state for Chapter {chapter_data.chapter_number}: {e}")
            return False

    async def _check_consistency(self, memories: List[MemoryCreate]) -> List:
        """Check for consistency issues in memories"""
        # This is a placeholder for consistency checking
        # In a full implementation, this would check for contradictions
        # between memories, timeline issues, etc.
        return []

    async def bulk_create_memories(self, bulk_request: BulkMemoryCreate) -> BulkMemoryResponse:
        """Create multiple memories in bulk"""
        created_ids = []
        errors = []
        created_count = 0

        try:
            # Add embeddings to all memories
            embeddings = await self._add_embeddings_to_memories(bulk_request.memories)

            async with self.db_manager.get_connection() as conn:
                async with conn.transaction():
                    for i, memory in enumerate(bulk_request.memories):
                        try:
                            embedding = embeddings[i] if i < len(embeddings) else None
                            memory_id = await self._create_new_memory(conn, memory, embedding)
                            created_ids.append(memory_id)
                            created_count += 1
                        except Exception as e:
                            errors.append(f"Failed to create memory: {str(e)}")

            return BulkMemoryResponse(
                created_count=created_count,
                failed_count=len(errors),
                created_ids=created_ids,
                errors=errors
            )

        except Exception as e:
            logger.error(f"Bulk memory creation failed: {e}")
            return BulkMemoryResponse(
                created_count=0,
                failed_count=len(bulk_request.memories),
                created_ids=[],
                errors=[str(e)]
            )