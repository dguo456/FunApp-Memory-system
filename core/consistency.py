from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime, timedelta
import logging

from core.schemas import (
    Memory, ConsistencyCheck, MemoryRelationship, RelationshipType,
    MemoryRelationshipCreate, ConsistencyLedgerCreate, ChangeType
)
from app.deps import DatabaseManager, EmbeddingService
import asyncpg

logger = logging.getLogger(__name__)


class MemoryConsistencyChecker:
    """Check for consistency issues across memories"""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service

        # Rules for detecting contradictions
        self.contradiction_patterns = [
            # Relationship status contradictions
            (r'\b(single|not in relationship|not dating)\b', r'\b(dating|relationship with|partner)\b'),
            (r'\b(trusting|happy with)\b', r'\b(suspicious|angry with|betrayed by)\b'),
            (r'\b(secret|hidden|concealed)\b', r'\b(public|open|revealed)\b'),

            # Timeline contradictions
            (r'\bfirst time\b', r'\balready\b|\bpreviously\b|\bbefore\b'),
            (r'\bnever\b', r'\balways\b|\busually\b'),

            # Location contradictions
            (r'\bat office\b', r'\bat home\b|\bat restaurant\b|\bat hotel\b'),
            (r'\balone\b', r'\btogether\b|\bwith\b'),

            # Emotional state contradictions
            (r'\bhappy\b|\bjoyful\b|\bexcited\b', r'\bsad\b|\bangry\b|\bdisappointed\b'),
            (r'\btrusting\b|\bconfident\b', r'\bsuspicious\b|\bdoubtful\b|\bworried\b')
        ]

    async def check_all_consistency(self) -> List[ConsistencyCheck]:
        """Check consistency across all memories"""
        try:
            all_issues = []

            # Check for contradictions
            contradictions = await self._check_contradictions()
            all_issues.extend(contradictions)

            # Check for timeline issues
            timeline_issues = await self._check_timeline_consistency()
            all_issues.extend(timeline_issues)

            # Check for relationship consistency
            relationship_issues = await self._check_relationship_consistency()
            all_issues.extend(relationship_issues)

            logger.info(f"Consistency check found {len(all_issues)} issues")
            return all_issues

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return []

    async def _check_contradictions(self) -> List[ConsistencyCheck]:
        """Check for contradictory memories"""
        issues = []

        try:
            # Get all memories grouped by character and related entity
            async with self.db_manager.get_connection() as conn:
                # Group memories by character and related entity
                query = """
                SELECT character_id, related_entity_id, memory_type,
                       array_agg(id ORDER BY chapter_number) as memory_ids,
                       array_agg(content ORDER BY chapter_number) as contents,
                       array_agg(chapter_number ORDER BY chapter_number) as chapters
                FROM memories
                WHERE related_entity_id IS NOT NULL
                GROUP BY character_id, related_entity_id, memory_type
                """

                rows = await conn.fetch(query)

                for row in rows:
                    memory_ids = row['memory_ids']
                    contents = row['contents']
                    chapters = row['chapters']

                    # Check for contradictions within this group
                    contradictions = self._find_contradictions_in_group(
                        memory_ids, contents, chapters
                    )
                    issues.extend(contradictions)

            return issues

        except Exception as e:
            logger.error(f"Failed to check contradictions: {e}")
            return []

    def _find_contradictions_in_group(
        self,
        memory_ids: List[int],
        contents: List[str],
        chapters: List[int]
    ) -> List[ConsistencyCheck]:
        """Find contradictions within a group of related memories"""
        contradictions = []

        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                content1, content2 = contents[i], contents[j]
                memory_id1, memory_id2 = memory_ids[i], memory_ids[j]

                # Check if these contents contradict each other
                is_contradiction, explanation = self._detect_contradiction(content1, content2)

                if is_contradiction:
                    consistency_score = self._calculate_contradiction_severity(content1, content2)

                    contradiction = ConsistencyCheck(
                        memory_id=memory_id1,
                        conflicting_memory_ids=[memory_id2],
                        consistency_score=consistency_score,
                        explanation=f"Contradiction between chapters {chapters[i]} and {chapters[j]}: {explanation}"
                    )
                    contradictions.append(contradiction)

        return contradictions

    def _detect_contradiction(self, content1: str, content2: str) -> Tuple[bool, str]:
        """Detect if two pieces of content contradict each other"""
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        for pattern1, pattern2 in self.contradiction_patterns:
            # Check if one content matches pattern1 and the other matches pattern2
            if (re.search(pattern1, content1_lower) and re.search(pattern2, content2_lower)) or \
               (re.search(pattern2, content1_lower) and re.search(pattern1, content2_lower)):
                explanation = f"Found contradictory patterns: '{pattern1}' vs '{pattern2}'"
                return True, explanation

        # Check for explicit negations of similar content
        if self._has_negation_contradiction(content1_lower, content2_lower):
            return True, "Found negation-based contradiction"

        return False, ""

    def _has_negation_contradiction(self, content1: str, content2: str) -> bool:
        """Check for negation-based contradictions"""
        # Extract key phrases and check for negations
        negation_words = ['not', 'never', 'no', 'none', 'neither', 'without']

        # Simple check for opposite statements
        for neg in negation_words:
            if neg in content1 and neg not in content2:
                # Look for similar key terms
                words1 = set(content1.split())
                words2 = set(content2.split())
                common_words = words1.intersection(words2)
                if len(common_words) > 3:  # Arbitrary threshold
                    return True

        return False

    def _calculate_contradiction_severity(self, content1: str, content2: str) -> float:
        """Calculate severity of contradiction (0.0 = no contradiction, 1.0 = severe)"""
        # This is a simplified scoring system
        severity = 0.0

        # Check for strong contradiction indicators
        strong_indicators = ['betrayal', 'lied', 'deceived', 'secret', 'hidden']
        for indicator in strong_indicators:
            if indicator in content1.lower() or indicator in content2.lower():
                severity += 0.3

        # Check for emotional contradictions
        positive_emotions = ['happy', 'trusting', 'loving', 'content']
        negative_emotions = ['angry', 'suspicious', 'hurt', 'disappointed']

        has_positive = any(emotion in content1.lower() or emotion in content2.lower() for emotion in positive_emotions)
        has_negative = any(emotion in content1.lower() or emotion in content2.lower() for emotion in negative_emotions)

        if has_positive and has_negative:
            severity += 0.4

        return min(severity, 1.0)

    async def _check_timeline_consistency(self) -> List[ConsistencyCheck]:
        """Check for timeline consistency issues"""
        issues = []

        try:
            # Check for memories that should follow a logical sequence
            async with self.db_manager.get_connection() as conn:
                query = """
                SELECT character_id, memory_type, related_entity_id,
                       array_agg(id ORDER BY chapter_number) as memory_ids,
                       array_agg(content ORDER BY chapter_number) as contents,
                       array_agg(chapter_number ORDER BY chapter_number) as chapters
                FROM memories
                GROUP BY character_id, memory_type, related_entity_id
                HAVING count(*) > 1
                """

                rows = await conn.fetch(query)

                for row in rows:
                    timeline_issues = self._check_timeline_sequence(
                        row['memory_ids'], row['contents'], row['chapters']
                    )
                    issues.extend(timeline_issues)

            return issues

        except Exception as e:
            logger.error(f"Failed to check timeline consistency: {e}")
            return []

    def _check_timeline_sequence(
        self,
        memory_ids: List[int],
        contents: List[str],
        chapters: List[int]
    ) -> List[ConsistencyCheck]:
        """Check if memories follow a logical timeline"""
        issues = []

        # Check for timeline markers that don't make sense
        timeline_markers = {
            'first': 1,
            'second': 2,
            'again': 2,
            'finally': 3,
            'last': 3
        }

        for i, content in enumerate(contents):
            content_lower = content.lower()
            for marker, expected_position in timeline_markers.items():
                if marker in content_lower:
                    # Check if this marker appears in the right position
                    relative_position = (i + 1) / len(contents)
                    expected_relative = expected_position / 3.0

                    if abs(relative_position - expected_relative) > 0.4:  # Threshold for timeline issues
                        issue = ConsistencyCheck(
                            memory_id=memory_ids[i],
                            conflicting_memory_ids=[],
                            consistency_score=0.6,
                            explanation=f"Timeline marker '{marker}' appears in unexpected position (chapter {chapters[i]})"
                        )
                        issues.append(issue)

        return issues

    async def _check_relationship_consistency(self) -> List[ConsistencyCheck]:
        """Check for consistency in relationship descriptions"""
        issues = []

        try:
            # Check C2U and IC memories for the same character pairs
            async with self.db_manager.get_connection() as conn:
                # Find character pairs that appear in both C2U and IC memories
                query = """
                WITH character_pairs AS (
                    SELECT DISTINCT
                        LEAST(character_id, related_entity_id) as char1,
                        GREATEST(character_id, related_entity_id) as char2
                    FROM memories
                    WHERE memory_type IN ('C2U', 'IC') AND related_entity_id IS NOT NULL
                )
                SELECT cp.char1, cp.char2,
                       array_agg(m.id) as memory_ids,
                       array_agg(m.content) as contents,
                       array_agg(m.memory_type) as types,
                       array_agg(m.character_id) as characters
                FROM character_pairs cp
                JOIN memories m ON (
                    (m.character_id = cp.char1 AND m.related_entity_id = cp.char2) OR
                    (m.character_id = cp.char2 AND m.related_entity_id = cp.char1)
                )
                WHERE m.memory_type IN ('C2U', 'IC')
                GROUP BY cp.char1, cp.char2
                HAVING count(*) > 1
                """

                rows = await conn.fetch(query)

                for row in rows:
                    relationship_issues = self._check_relationship_perspective_consistency(
                        row['memory_ids'], row['contents'], row['types'], row['characters']
                    )
                    issues.extend(relationship_issues)

            return issues

        except Exception as e:
            logger.error(f"Failed to check relationship consistency: {e}")
            return []

    def _check_relationship_perspective_consistency(
        self,
        memory_ids: List[int],
        contents: List[str],
        types: List[str],
        characters: List[int]
    ) -> List[ConsistencyCheck]:
        """Check if different perspectives on the same relationship are consistent"""
        issues = []

        # Group memories by perspective (character who remembers)
        perspectives = {}
        for i, char_id in enumerate(characters):
            if char_id not in perspectives:
                perspectives[char_id] = []
            perspectives[char_id].append({
                'memory_id': memory_ids[i],
                'content': contents[i],
                'type': types[i]
            })

        # Compare perspectives
        perspective_list = list(perspectives.values())
        for i in range(len(perspective_list)):
            for j in range(i + 1, len(perspective_list)):
                persp1, persp2 = perspective_list[i], perspective_list[j]

                # Compare the relationship descriptions
                inconsistency_score = self._compare_relationship_perspectives(persp1, persp2)

                if inconsistency_score > 0.7:  # Threshold for significant inconsistency
                    # Find the most recent memory from each perspective
                    recent1 = max(persp1, key=lambda x: x['memory_id'])
                    recent2 = max(persp2, key=lambda x: x['memory_id'])

                    issue = ConsistencyCheck(
                        memory_id=recent1['memory_id'],
                        conflicting_memory_ids=[recent2['memory_id']],
                        consistency_score=1.0 - inconsistency_score,
                        explanation=f"Inconsistent relationship perspectives between characters"
                    )
                    issues.append(issue)

        return issues

    def _compare_relationship_perspectives(self, persp1: List[dict], persp2: List[dict]) -> float:
        """Compare two relationship perspectives and return inconsistency score"""
        # This is a simplified comparison
        # In a full implementation, this would use more sophisticated NLP

        content1 = " ".join([p['content'] for p in persp1]).lower()
        content2 = " ".join([p['content'] for p in persp2]).lower()

        # Check for contradictory relationship descriptors
        positive_descriptors = ['love', 'trust', 'happy', 'close', 'intimate']
        negative_descriptors = ['angry', 'suspicious', 'distant', 'betrayed', 'hurt']

        pos1 = sum(1 for desc in positive_descriptors if desc in content1)
        neg1 = sum(1 for desc in negative_descriptors if desc in content1)
        pos2 = sum(1 for desc in positive_descriptors if desc in content2)
        neg2 = sum(1 for desc in negative_descriptors if desc in content2)

        # If one perspective is mostly positive and the other mostly negative
        if (pos1 > neg1 and neg2 > pos2) or (neg1 > pos1 and pos2 > neg2):
            return 0.8

        return 0.0

    async def create_memory_relationship(
        self,
        source_memory_id: int,
        target_memory_id: int,
        relationship_type: RelationshipType,
        confidence_score: float = 1.0
    ) -> int:
        """Create a relationship between two memories"""
        try:
            async with self.db_manager.get_connection() as conn:
                relationship_id = await conn.fetchval(
                    """
                    INSERT INTO memory_relationships (source_memory_id, target_memory_id, relationship_type, confidence_score)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                    """,
                    source_memory_id, target_memory_id, relationship_type.value, confidence_score
                )

                logger.info(f"Created memory relationship {relationship_id}: {source_memory_id} -> {target_memory_id}")
                return relationship_id

        except Exception as e:
            logger.error(f"Failed to create memory relationship: {e}")
            raise

    async def flag_memory_for_review(self, memory_id: int, reason: str):
        """Flag a memory for manual review due to consistency issues"""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO consistency_ledger (memory_id, change_type, change_reason, flagged_for_review)
                    VALUES ($1, $2, $3, $4)
                    """,
                    memory_id, ChangeType.FLAGGED.value, reason, True
                )

                logger.info(f"Flagged memory {memory_id} for review: {reason}")

        except Exception as e:
            logger.error(f"Failed to flag memory {memory_id}: {e}")

    async def get_flagged_memories(self) -> List[int]:
        """Get all memories flagged for review"""
        try:
            async with self.db_manager.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT memory_id
                    FROM consistency_ledger
                    WHERE flagged_for_review = true
                    """
                )
                return [row['memory_id'] for row in rows]

        except Exception as e:
            logger.error(f"Failed to get flagged memories: {e}")
            return []