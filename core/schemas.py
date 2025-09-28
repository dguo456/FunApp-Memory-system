from datetime import datetime
from typing import List, Optional, Tuple, Dict
from enum import Enum
from pydantic import BaseModel, Field, validator


class MemoryType(str, Enum):
    CHARACTER_TO_USER = "C2U"
    INTER_CHARACTER = "IC"
    WORLD_MEMORY = "WM"


class RelationshipType(str, Enum):
    UPDATES = "UPDATES"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    REFERENCES = "REFERENCES"


class ChangeType(str, Enum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    MERGED = "MERGED"
    FLAGGED = "FLAGGED"


# Base schemas
class CharacterBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None


class CharacterCreate(CharacterBase):
    pass


class Character(CharacterBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    name: str = Field(..., max_length=100)


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class WorldStateBase(BaseModel):
    chapter_number: int
    state_description: str


class WorldStateCreate(WorldStateBase):
    pass


class WorldState(WorldStateBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Memory schemas
class MemoryBase(BaseModel):
    memory_type: MemoryType
    character_id: int
    related_entity_id: Optional[int] = None
    content: str
    summary: Optional[str] = None
    chapter_number: Optional[int] = None
    context_tags: List[str] = Field(default_factory=list)
    importance_score: float = Field(default=1.0, ge=0.0, le=10.0)

    @validator('related_entity_id')
    def validate_related_entity_id(cls, v, values):
        memory_type = values.get('memory_type')
        if memory_type == MemoryType.CHARACTER_TO_USER and v is None:
            raise ValueError("related_entity_id is required for Character-to-User memories")
        if memory_type == MemoryType.INTER_CHARACTER and v is None:
            raise ValueError("related_entity_id is required for Inter-Character memories")
        return v


class MemoryCreate(MemoryBase):
    pass


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    summary: Optional[str] = None
    context_tags: Optional[List[str]] = None
    importance_score: Optional[float] = Field(None, ge=0.0, le=10.0)


class Memory(MemoryBase):
    id: int
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Memory relationship schemas
class MemoryRelationshipBase(BaseModel):
    source_memory_id: int
    target_memory_id: int
    relationship_type: RelationshipType
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class MemoryRelationshipCreate(MemoryRelationshipBase):
    pass


class MemoryRelationship(MemoryRelationshipBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Consistency ledger schemas
class ConsistencyLedgerBase(BaseModel):
    memory_id: int
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    change_reason: Optional[str] = None
    flagged_for_review: bool = False


class ConsistencyLedgerCreate(ConsistencyLedgerBase):
    pass


class ConsistencyLedger(ConsistencyLedgerBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Query and retrieval schemas
class MemoryQuery(BaseModel):
    query_text: str
    memory_types: Optional[List[MemoryType]] = None
    character_ids: Optional[List[int]] = None
    chapter_range: Optional[Tuple[int, int]] = None
    context_tags: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)  # Weight for semantic vs keyword search


class MemorySearchResult(BaseModel):
    memory: Memory
    similarity_score: float
    rank: int
    retrieval_method: str  # "semantic", "keyword", "hybrid"


class RetrievalResponse(BaseModel):
    query: str
    results: List[MemorySearchResult]
    total_found: int
    query_embedding: Optional[List[float]] = None


# Evaluation schemas
class EvaluationQuery(BaseModel):
    query_text: str
    expected_memory_ids: List[int]
    query_type: str  # "precision", "recall", "consistency"
    context: Optional[dict] = None


class EvaluationResult(BaseModel):
    query_id: str
    precision_at_k: Dict[int, float]  # P@1, P@5, P@10
    recall_at_k: Dict[int, float]     # R@1, R@5, R@10
    mrr: float  # Mean Reciprocal Rank
    found_memories: List[int]
    missing_memories: List[int]
    consistency_score: Optional[float] = None


class ConsistencyCheck(BaseModel):
    memory_id: int
    conflicting_memory_ids: List[int]
    consistency_score: float
    explanation: str


# Bulk operations
class BulkMemoryCreate(BaseModel):
    memories: List[MemoryCreate]
    batch_size: int = Field(default=100, ge=1, le=1000)


class BulkMemoryResponse(BaseModel):
    created_count: int
    failed_count: int
    created_ids: List[int]
    errors: List[str]


# Chapter processing
class ChapterData(BaseModel):
    chapter_number: int
    synopsis: str
    characters_involved: Optional[List[str]] = None
    extracted_memories: Optional[List[MemoryCreate]] = None


class ChapterProcessingResult(BaseModel):
    chapter_number: int
    memories_created: int
    memories_updated: int
    world_state_updated: bool
    consistency_issues: List[ConsistencyCheck]


# Analytics and metrics
class MemoryBasic(BaseModel):
    """Basic memory info for stats without embeddings"""
    id: int
    content: str
    summary: Optional[str] = None
    access_count: Optional[int] = None
    created_at: Optional[datetime] = None

class MemoryStats(BaseModel):
    total_memories: int
    memories_by_type: Dict[MemoryType, int]
    memories_by_character: Dict[str, int]
    avg_importance_score: float
    most_accessed_memories: List[MemoryBasic]
    recent_memories: List[MemoryBasic]


class SystemHealth(BaseModel):
    database_status: str
    embedding_service_status: str
    total_memories: int
    flagged_for_review: int
    last_update: datetime
    consistency_score: float