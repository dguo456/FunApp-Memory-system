-- Sekai Memory System Database Schema
-- Supports three types of memories: Character-to-User (C2U), Inter-Character (IC), World Memory (WM)

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Characters table
CREATE TABLE characters (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table (for the user character)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- World state table to track world changes across chapters
CREATE TABLE world_states (
    id SERIAL PRIMARY KEY,
    chapter_number INTEGER NOT NULL,
    state_description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Main memories table supporting all three memory types
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('C2U', 'IC', 'WM')),

    -- For C2U: character_id = remembering character, related_entity_id = user_id
    -- For IC: character_id = remembering character, related_entity_id = other character_id
    -- For WM: character_id = remembering character, related_entity_id = NULL (or world_state_id)
    character_id INTEGER NOT NULL REFERENCES characters(id),
    related_entity_id INTEGER, -- Can reference users(id) or characters(id) depending on memory_type

    -- Memory content
    content TEXT NOT NULL,
    summary TEXT, -- Brief summary for fast lookup

    -- Context information
    chapter_number INTEGER,
    context_tags TEXT[], -- Array of tags for categorization

    -- Vector embeddings for semantic search
    embedding vector(384), -- all-MiniLM-L6-v2 produces 384-dimensional vectors

    -- Memory strength and importance
    importance_score FLOAT DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory relationships to track memory dependencies and updates
CREATE TABLE memory_relationships (
    id SERIAL PRIMARY KEY,
    source_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL CHECK (relationship_type IN ('UPDATES', 'CONTRADICTS', 'SUPPORTS', 'REFERENCES')),
    confidence_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Memory consistency ledger for tracking changes and potential conflicts
CREATE TABLE consistency_ledger (
    id SERIAL PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    change_type VARCHAR(20) NOT NULL CHECK (change_type IN ('CREATED', 'UPDATED', 'MERGED', 'FLAGGED')),
    old_content TEXT,
    new_content TEXT,
    change_reason TEXT,
    flagged_for_review BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance

-- Memory type and character indexes
CREATE INDEX idx_memories_type_character ON memories(memory_type, character_id);
CREATE INDEX idx_memories_chapter ON memories(chapter_number);
CREATE INDEX idx_memories_importance ON memories(importance_score DESC);
CREATE INDEX idx_memories_accessed ON memories(last_accessed DESC);

-- Vector similarity index (HNSW for fast approximate nearest neighbor search)
CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);

-- Full-text search indexes for sparse retrieval
CREATE INDEX idx_memories_content_fts ON memories USING gin(to_tsvector('english', content));
CREATE INDEX idx_memories_summary_fts ON memories USING gin(to_tsvector('english', summary));

-- Tags index for fast tag-based filtering
CREATE INDEX idx_memories_tags ON memories USING gin(context_tags);

-- Relationship indexes
CREATE INDEX idx_memory_relationships_source ON memory_relationships(source_memory_id);
CREATE INDEX idx_memory_relationships_target ON memory_relationships(target_memory_id);
CREATE INDEX idx_memory_relationships_type ON memory_relationships(relationship_type);

-- Consistency ledger indexes
CREATE INDEX idx_consistency_ledger_memory ON consistency_ledger(memory_id);
CREATE INDEX idx_consistency_ledger_flagged ON consistency_ledger(flagged_for_review);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_characters_updated_at BEFORE UPDATE ON characters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to increment access count
CREATE OR REPLACE FUNCTION increment_memory_access(memory_id INTEGER)
RETURNS VOID AS $$
BEGIN
    UPDATE memories
    SET access_count = access_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE id = memory_id;
END;
$$ LANGUAGE plpgsql;

-- Sample data insertion for testing
INSERT INTO characters (name, description) VALUES
    ('Byleth', 'Strategic and calculating main character'),
    ('Dimitri', 'Intense and focused character'),
    ('Sylvain', 'Charming and flirtatious character'),
    ('Annette', 'Cheerful and trusting character'),
    ('Felix', 'Sharp and analytical character'),
    ('Dedue', 'Loyal and observant character'),
    ('Mercedes', 'Kind and supportive character'),
    ('Ashe', 'Gentle and concerned character');

INSERT INTO users (name) VALUES ('User');

-- Insert initial world state
INSERT INTO world_states (chapter_number, state_description) VALUES
    (1, 'Normal corporate office environment at Garreg Mach Corp');