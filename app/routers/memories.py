from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
import logging

from core.schemas import (
    Memory, MemoryCreate, MemoryUpdate, MemoryQuery,
    RetrievalResponse, MemoryType, Character, User,
    BulkMemoryCreate, BulkMemoryResponse
)
from core.retrieval import HybridMemoryRetriever
from core.ingest import MemoryIngestionService
from app.deps import db_manager, embedding_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_retriever(request: Request) -> HybridMemoryRetriever:
    """Get retriever from app state"""
    return request.app.state.retriever


def get_ingestion_service(request: Request) -> MemoryIngestionService:
    """Get ingestion service from app state"""
    return request.app.state.ingestion_service


@router.post("/memories", response_model=Memory)
async def create_memory(
    memory_data: MemoryCreate,
    ingestion_service: MemoryIngestionService = Depends(get_ingestion_service)
) -> Memory:
    """Create a new memory"""
    try:
        # Add embedding to the memory
        await ingestion_service._add_embeddings_to_memories([memory_data])

        # Store the memory
        async with db_manager.get_connection() as conn:
            memory_id = await ingestion_service._create_new_memory(conn, memory_data)

            # Retrieve the created memory
            row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)

            if not row:
                raise HTTPException(status_code=500, detail="Failed to retrieve created memory")

            # Convert to Memory object
            memory = Memory(
                id=row['id'],
                memory_type=MemoryType(row['memory_type']),
                character_id=row['character_id'],
                related_entity_id=row['related_entity_id'],
                content=row['content'],
                summary=row['summary'],
                chapter_number=row['chapter_number'],
                context_tags=list(row['context_tags']) if row['context_tags'] else [],
                embedding=list(row['embedding']) if row['embedding'] else None,
                importance_score=float(row['importance_score']),
                access_count=row['access_count'],
                last_accessed=row['last_accessed'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

            return memory

    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")


@router.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(
    memory_id: int,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
) -> Memory:
    """Get a specific memory by ID"""
    memory = await retriever.get_memory_by_id(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.put("/memories/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: int,
    memory_update: MemoryUpdate,
    ingestion_service: MemoryIngestionService = Depends(get_ingestion_service)
) -> Memory:
    """Update an existing memory"""
    try:
        async with db_manager.get_connection() as conn:
            # Check if memory exists
            existing = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Memory not found")

            # Build update query dynamically
            update_fields = []
            params = []
            param_count = 0

            if memory_update.content is not None:
                param_count += 1
                update_fields.append(f"content = ${param_count}")
                params.append(memory_update.content)

                # Generate new embedding if content changed
                if memory_update.content != existing['content']:
                    new_embedding = embedding_service.encode_text(memory_update.content)
                    param_count += 1
                    update_fields.append(f"embedding = ${param_count}")
                    params.append(new_embedding.tolist())

            if memory_update.summary is not None:
                param_count += 1
                update_fields.append(f"summary = ${param_count}")
                params.append(memory_update.summary)

            if memory_update.context_tags is not None:
                param_count += 1
                update_fields.append(f"context_tags = ${param_count}")
                params.append(memory_update.context_tags)

            if memory_update.importance_score is not None:
                param_count += 1
                update_fields.append(f"importance_score = ${param_count}")
                params.append(memory_update.importance_score)

            if not update_fields:
                raise HTTPException(status_code=400, detail="No fields to update")

            # Add updated_at
            update_fields.append("updated_at = CURRENT_TIMESTAMP")

            # Execute update
            param_count += 1
            query = f"UPDATE memories SET {', '.join(update_fields)} WHERE id = ${param_count}"
            params.append(memory_id)

            await conn.execute(query, *params)

            # Log the update
            await ingestion_service._log_consistency_change(
                conn, memory_id, "UPDATED",
                existing['content'],
                memory_update.content or existing['content'],
                "Manual update via API"
            )

            # Return updated memory
            updated_row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)

            return Memory(
                id=updated_row['id'],
                memory_type=MemoryType(updated_row['memory_type']),
                character_id=updated_row['character_id'],
                related_entity_id=updated_row['related_entity_id'],
                content=updated_row['content'],
                summary=updated_row['summary'],
                chapter_number=updated_row['chapter_number'],
                context_tags=list(updated_row['context_tags']) if updated_row['context_tags'] else [],
                embedding=list(updated_row['embedding']) if updated_row['embedding'] else None,
                importance_score=float(updated_row['importance_score']),
                access_count=updated_row['access_count'],
                last_accessed=updated_row['last_accessed'],
                created_at=updated_row['created_at'],
                updated_at=updated_row['updated_at']
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int):
    """Delete a memory"""
    try:
        async with db_manager.get_connection() as conn:
            # Check if memory exists
            existing = await conn.fetchrow("SELECT id FROM memories WHERE id = $1", memory_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Memory not found")

            # Delete the memory (relationships will be cascade deleted)
            await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

            return {"message": f"Memory {memory_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


@router.post("/memories/search", response_model=RetrievalResponse)
async def search_memories(
    query: MemoryQuery,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
) -> RetrievalResponse:
    """Search memories using hybrid retrieval"""
    try:
        return await retriever.search_memories(query)
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/memories/{memory_id}/related")
async def get_related_memories(
    memory_id: int,
    limit: int = 10,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
):
    """Get memories related to a specific memory"""
    try:
        related = await retriever.get_related_memories(memory_id, limit)
        return {"memory_id": memory_id, "related_memories": related}
    except Exception as e:
        logger.error(f"Failed to get related memories for {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get related memories: {str(e)}")


@router.get("/characters/{character_id}/memories", response_model=List[Memory])
async def get_character_memories(
    character_id: int,
    memory_type: Optional[MemoryType] = None,
    limit: int = 50,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
) -> List[Memory]:
    """Get all memories for a specific character"""
    try:
        return await retriever.get_character_memories(character_id, memory_type, limit)
    except Exception as e:
        logger.error(f"Failed to get character memories for {character_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get character memories: {str(e)}")


@router.post("/memories/bulk", response_model=BulkMemoryResponse)
async def create_bulk_memories(
    bulk_request: BulkMemoryCreate,
    ingestion_service: MemoryIngestionService = Depends(get_ingestion_service)
) -> BulkMemoryResponse:
    """Create multiple memories in bulk"""
    try:
        return await ingestion_service.bulk_create_memories(bulk_request)
    except Exception as e:
        logger.error(f"Bulk memory creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk creation failed: {str(e)}")


@router.get("/characters", response_model=List[Character])
async def get_characters():
    """Get all characters"""
    try:
        async with db_manager.get_connection() as conn:
            rows = await conn.fetch("SELECT * FROM characters ORDER BY name")

            characters = []
            for row in rows:
                character = Character(
                    id=row['id'],
                    name=row['name'],
                    description=row['description'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                characters.append(character)

            return characters

    except Exception as e:
        logger.error(f"Failed to get characters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get characters: {str(e)}")


@router.get("/users", response_model=List[User])
async def get_users():
    """Get all users"""
    try:
        async with db_manager.get_connection() as conn:
            rows = await conn.fetch("SELECT * FROM users ORDER BY name")

            users = []
            for row in rows:
                user = User(
                    id=row['id'],
                    name=row['name'],
                    created_at=row['created_at']
                )
                users.append(user)

            return users

    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get users: {str(e)}")


@router.post("/memories/search/context")
async def search_by_context(
    character_id: int,
    context_tags: List[str],
    chapter_start: Optional[int] = None,
    chapter_end: Optional[int] = None,
    limit: int = 20,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
):
    """Search memories by context tags"""
    try:
        chapter_range = None
        if chapter_start is not None and chapter_end is not None:
            chapter_range = (chapter_start, chapter_end)

        memories = await retriever.search_by_context(
            character_id, context_tags, chapter_range, limit
        )

        return {
            "character_id": character_id,
            "context_tags": context_tags,
            "chapter_range": chapter_range,
            "memories": memories
        }

    except Exception as e:
        logger.error(f"Context search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context search failed: {str(e)}")