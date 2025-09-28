import re
from typing import List, Dict, Optional, Tuple
from core.schemas import (
    MemoryCreate, MemoryType, ChapterData,
    Character, User, ConsistencyLedgerCreate, ChangeType
)
import logging

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """Extract memories from chapter synopses using rule-based and optional LLM approaches"""

    def __init__(self):
        # Character name patterns and relationships
        self.character_patterns = {
            'Byleth': r'\b[Bb]yleth\b',
            'Dimitri': r'\b[Dd]imitri\b',
            'Sylvain': r'\b[Ss]ylvain\b',
            'Annette': r'\b[Aa]nnette\b',
            'Felix': r'\b[Ff]elix\b',
            'Dedue': r'\b[Dd]edue\b',
            'Mercedes': r'\b[Mm]ercedes\b',
            'Ashe': r'\b[Aa]she\b'
        }

        # Relationship indicators
        self.relationship_patterns = {
            'romantic': r'\b(kiss|passion|intimate|affair|captivated|charm|flirtation)\b',
            'professional': r'\b(office|work|desk|meeting|corporate|task|presentation)\b',
            'secretive': r'\b(secret|discreet|hidden|private|cover|concealed|covert)\b',
            'emotional': r'\b(anger|happy|disappointed|trust|concern|worry)\b',
            'physical': r'\b(touch|embrace|hand|closeness|proximity)\b'
        }

        # Memory importance keywords
        self.importance_keywords = {
            'high': r'\b(betrayal|affair|confrontation|conflict|discovery|evidence)\b',
            'medium': r'\b(suspicion|doubt|trust|relationship|plan|strategy)\b',
            'low': r'\b(observation|note|mention|casual|brief)\b'
        }

    def extract_memories_from_chapter(
        self,
        chapter_data: ChapterData,
        characters: Dict[str, Character],
        user: User
    ) -> List[MemoryCreate]:
        """Extract memories from a single chapter synopsis"""
        memories = []
        synopsis = chapter_data.synopsis
        chapter_num = chapter_data.chapter_number

        # Extract character mentions and their contexts
        character_contexts = self._extract_character_contexts(synopsis)

        # Generate memories for each character mentioned
        for char_name, context in character_contexts.items():
            if char_name not in characters:
                continue

            character = characters[char_name]

            # Generate different types of memories based on context
            memories.extend(self._generate_character_memories(
                character, synopsis, context, chapter_num, characters, user
            ))

        # Extract world state memories
        world_memories = self._extract_world_memories(synopsis, chapter_num, characters)
        memories.extend(world_memories)

        return memories

    def _extract_character_contexts(self, synopsis: str) -> Dict[str, str]:
        """Extract context around each character mention"""
        contexts = {}

        for char_name, pattern in self.character_patterns.items():
            matches = list(re.finditer(pattern, synopsis, re.IGNORECASE))
            if matches:
                # Get context around the character mention
                context_parts = []
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(synopsis), match.end() + 50)
                    context_parts.append(synopsis[start:end])

                contexts[char_name] = ' '.join(context_parts)

        return contexts

    def _generate_character_memories(
        self,
        character: Character,
        synopsis: str,
        context: str,
        chapter_num: int,
        characters: Dict[str, Character],
        user: User
    ) -> List[MemoryCreate]:
        """Generate memories for a specific character"""
        memories = []

        # Character-to-User memories (from character's perspective about user/Byleth)
        if character.name != 'Byleth':
            c2u_memory = self._create_c2u_memory(
                character, synopsis, context, chapter_num, user
            )
            if c2u_memory:
                memories.append(c2u_memory)

        # Inter-character memories
        ic_memories = self._create_inter_character_memories(
            character, synopsis, context, chapter_num, characters
        )
        memories.extend(ic_memories)

        # World memories for this character
        wm_memory = self._create_world_memory(
            character, synopsis, context, chapter_num
        )
        if wm_memory:
            memories.append(wm_memory)

        return memories

    def _create_c2u_memory(
        self,
        character: Character,
        synopsis: str,
        context: str,
        chapter_num: int,
        user: User
    ) -> Optional[MemoryCreate]:
        """Create Character-to-User memory"""

        # Extract character's perspective on Byleth/User
        if 'Byleth' not in synopsis:
            return None

        # Determine the nature of the interaction
        relationship_type = self._determine_relationship_type(context)
        importance = self._calculate_importance(context)

        # Create memory content from character's perspective
        if character.name == 'Dimitri':
            if 'kiss' in synopsis.lower() or 'passion' in synopsis.lower():
                content = f"I shared an intimate moment with Byleth in my office. The professional boundary was crossed, and I find myself deeply attracted to them."
            elif 'dinner' in synopsis.lower() and 'apartment' in synopsis.lower():
                content = f"I invited Byleth to my apartment for dinner. Our connection is deepening beyond the workplace."
            elif 'lunch' in synopsis.lower() and 'hand' in synopsis.lower():
                content = f"Had lunch with Byleth at La Maison. I couldn't help but show my affection publicly by taking their hand."
            else:
                content = f"Had an interaction with Byleth during Chapter {chapter_num}. {context[:100]}..."

        elif character.name == 'Sylvain':
            if 'hotel' in synopsis.lower() or 'bar' in synopsis.lower():
                content = f"Spent passionate time with Byleth away from the office. The secrecy makes it even more thrilling."
            elif 'angry' in synopsis.lower() or 'confrontation' in synopsis.lower():
                content = f"Confronted Byleth about their relationship with Dimitri. I felt betrayed seeing them together so openly."
            else:
                content = f"Had contact with Byleth in Chapter {chapter_num}. They continue to intrigue me."

        elif character.name == 'Dedue':
            if 'earring' in synopsis.lower():
                content = f"Found evidence of Byleth's presence in Dimitri's apartment. I'm concerned about this workplace entanglement."
            elif 'warning' in synopsis.lower():
                content = f"Warned Dimitri about his involvement with Byleth. These relationships could lead to complications."
            else:
                content = f"Observed Byleth's behavior around Dimitri. Something seems different about their dynamic."
        else:
            content = f"Interacted with Byleth during Chapter {chapter_num}. {self._extract_key_details(context)}"

        tags = [relationship_type, f"chapter_{chapter_num}"]

        return MemoryCreate(
            memory_type=MemoryType.CHARACTER_TO_USER,
            character_id=character.id,
            related_entity_id=user.id,
            content=content,
            summary=f"{character.name}'s interaction with Byleth in Chapter {chapter_num}",
            chapter_number=chapter_num,
            context_tags=tags,
            importance_score=importance
        )

    def _create_inter_character_memories(
        self,
        character: Character,
        synopsis: str,
        context: str,
        chapter_num: int,
        characters: Dict[str, Character]
    ) -> List[MemoryCreate]:
        """Create Inter-Character memories"""
        memories = []

        # Find other characters mentioned in the same context
        for other_name, other_char in characters.items():
            if other_name == character.name:
                continue

            if other_name.lower() in synopsis.lower():
                # Create memory about the other character
                relationship_type = self._determine_relationship_type(context)
                importance = self._calculate_importance(context)

                content = self._generate_ic_content(
                    character.name, other_name, synopsis, chapter_num
                )

                if content:
                    memory = MemoryCreate(
                        memory_type=MemoryType.INTER_CHARACTER,
                        character_id=character.id,
                        related_entity_id=other_char.id,
                        content=content,
                        summary=f"{character.name}'s memory about {other_name} in Chapter {chapter_num}",
                        chapter_number=chapter_num,
                        context_tags=[relationship_type, f"chapter_{chapter_num}", f"involves_{other_name.lower()}"],
                        importance_score=importance
                    )
                    memories.append(memory)

        return memories

    def _create_world_memory(
        self,
        character: Character,
        synopsis: str,
        context: str,
        chapter_num: int
    ) -> Optional[MemoryCreate]:
        """Create World Memory for environmental/world state changes"""

        world_events = []

        # Detect significant world/environment changes
        if re.search(r'\b(office|corporate|building|workplace)\b', synopsis, re.IGNORECASE):
            world_events.append("corporate_environment")

        if re.search(r'\b(restaurant|bar|hotel|apartment)\b', synopsis, re.IGNORECASE):
            world_events.append("location_change")

        if re.search(r'\b(memo|alert|announcement|company-wide)\b', synopsis, re.IGNORECASE):
            world_events.append("company_communication")

        if not world_events:
            return None

        content = f"Chapter {chapter_num}: {self._extract_key_details(synopsis)}"
        importance = self._calculate_importance(synopsis)

        return MemoryCreate(
            memory_type=MemoryType.WORLD_MEMORY,
            character_id=character.id,
            related_entity_id=None,
            content=content,
            summary=f"World state as observed by {character.name} in Chapter {chapter_num}",
            chapter_number=chapter_num,
            context_tags=world_events + [f"chapter_{chapter_num}"],
            importance_score=importance
        )

    def _generate_ic_content(self, char1: str, char2: str, synopsis: str, chapter_num: int) -> Optional[str]:
        """Generate inter-character memory content"""

        # Specific relationship dynamics
        if char1 == 'Sylvain' and char2 == 'Annette':
            if 'surprise' in synopsis.lower() and 'weekend' in synopsis.lower():
                return f"Annette is planning a surprise romantic getaway for me. She seems so happy and trusting."
            elif 'disappointed' in synopsis.lower() or 'cancel' in synopsis.lower():
                return f"Had to cancel plans with Annette for work. She sounded disappointed but understanding."
            elif 'happy' in synopsis.lower():
                return f"Annette seems particularly happy lately. Our relationship feels strong."

        elif char1 == 'Annette' and char2 == 'Sylvain':
            if 'surprise' in synopsis.lower():
                return f"Planning a special surprise for Sylvain - a romantic countryside getaway."
            elif 'cancelled' in synopsis.lower() or 'work' in synopsis.lower():
                return f"Sylvain had to cancel our weekend plans for work. I'm disappointed but trying to be understanding."

        elif char1 == 'Dedue' and char2 == 'Dimitri':
            if 'warning' in synopsis.lower() or 'concern' in synopsis.lower():
                return f"I'm concerned about Dimitri's workplace relationship with Byleth. Warned him about potential complications."
            elif 'earring' in synopsis.lower():
                return f"Found evidence of Byleth's presence in Dimitri's apartment. This relationship is becoming more serious."

        elif char1 == 'Felix' and char2 == 'Byleth':
            if 'observing' in synopsis.lower() or 'calculating' in synopsis.lower():
                return f"Byleth's behavior is suspicious. They're hiding something and I can tell."

        # Generic interaction
        return f"Had an interaction with {char2} during Chapter {chapter_num}."

    def _determine_relationship_type(self, context: str) -> str:
        """Determine the type of relationship based on context"""
        for rel_type, pattern in self.relationship_patterns.items():
            if re.search(pattern, context, re.IGNORECASE):
                return rel_type
        return "general"

    def _calculate_importance(self, context: str) -> float:
        """Calculate importance score based on content"""
        for importance_level, pattern in self.importance_keywords.items():
            if re.search(pattern, context, re.IGNORECASE):
                if importance_level == 'high':
                    return 8.0
                elif importance_level == 'medium':
                    return 5.0
                else:
                    return 3.0
        return 1.0

    def _extract_key_details(self, text: str) -> str:
        """Extract key details from text for summary"""
        # Remove common words and extract meaningful phrases
        sentences = text.split('.')
        key_sentence = sentences[0] if sentences else text
        return key_sentence.strip()[:150] + "..." if len(key_sentence) > 150 else key_sentence.strip()

    def _extract_world_memories(self, synopsis: str, chapter_num: int, characters: Dict[str, Character]) -> List[MemoryCreate]:
        """Extract world state memories that apply to all characters"""
        memories = []

        # Check for significant world events mentioned
        if 'virus' in synopsis.lower() and 'health alert' in synopsis.lower():
            # Create world memory for the virus alert
            for char_name, character in characters.items():
                content = f"Company issued a health alert about a novel virus spreading overseas. Most colleagues dismissed it as distant concern."
                memory = MemoryCreate(
                    memory_type=MemoryType.WORLD_MEMORY,
                    character_id=character.id,
                    related_entity_id=None,
                    content=content,
                    summary=f"Health alert about virus outbreak - Chapter {chapter_num}",
                    chapter_number=chapter_num,
                    context_tags=["health_alert", "virus", "company_communication", f"chapter_{chapter_num}"],
                    importance_score=4.0
                )
                memories.append(memory)

        return memories