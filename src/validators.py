"""
Centralized validation logic for the Semantic Router training pipeline.
Use this module across generation, translation, and training layers.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.schemas import TrainingExample, AgentOutput
from src.config import VALIDATION_CONFIG
from src.infrastructure import logger


@dataclass
class ValidationResult:
    """Result of validation with detailed feedback."""
    is_valid: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class DataValidator:
    """
    Robust validation for training examples.
    Separates concerns: structural vs quality vs domain-specific checks.
    """
    
    # --- Configuration ---
    MIN_QUERY_LENGTH = VALIDATION_CONFIG["MIN_QUERY_LENGTH"]
    MIN_THOUGHT_WORDS = VALIDATION_CONFIG["MIN_THOUGHT_WORDS"]
    MAX_THOUGHT_WORDS = VALIDATION_CONFIG["MAX_THOUGHT_WORDS"]
    MIN_FINAL_ANSWER_LENGTH = VALIDATION_CONFIG["MIN_FINAL_ANSWER_LENGTH"]
    PARROTING_THRESHOLD = VALIDATION_CONFIG["PARROTING_THRESHOLD"]
    
    @staticmethod
    def validate_structural(item: TrainingExample) -> ValidationResult:
        """
        Layer 1: Structural.
        Most logic is now in src/schemas.py @model_validator.
        We just check for null object integrity here.
        """
        if not isinstance(item.output, AgentOutput):
             return ValidationResult(False, "structural", "Output is not AgentOutput")
        return ValidationResult(True)

    @classmethod
    def validate_quality(cls, item: TrainingExample) -> ValidationResult:
        """
        Layer 2: Quality validation (content quality, not just structure).
        This is where we check for good training data.
        """
        warnings = []
        
        # Check 1: Query Quality
        query = item.user_query.strip()
        if len(query) < cls.MIN_QUERY_LENGTH:
            return ValidationResult(
                is_valid=False,
                error_type="quality",
                error_message=f"Query too short: {len(query)} chars (min: {cls.MIN_QUERY_LENGTH})"
            )
        
        # Check for placeholder text
        if any(ph in query.lower() for ph in ["lorem ipsum", "test", "placeholder", "xxx"]):
            warnings.append("Query contains placeholder-like text")
        
        # Status-specific quality checks
        if item.output.status == "running":
            # Check 2: Thought Quality
            thought = item.output.thought.strip()
            word_count = len(thought.split())
            
            if word_count < cls.MIN_THOUGHT_WORDS:
                return ValidationResult(
                    is_valid=False,
                    error_type="quality",
                    error_message=f"Thought too short: {word_count} words (min: {cls.MIN_THOUGHT_WORDS})"
                )
            
            if word_count > cls.MAX_THOUGHT_WORDS:
                return ValidationResult(
                    is_valid=False,
                    error_type="quality",
                    error_message=f"Thought too long: {word_count} words (max: {cls.MAX_THOUGHT_WORDS})"
                )
            
            # Check 3: Parroting Detection
            if cls._is_parroting(query, thought):
                return ValidationResult(
                    is_valid=False,
                    error_type="quality",
                    error_message="Thought is parroting the query"
                )
            
            # Check 4: Generic thoughts
            generic_phrases = [
                "i need to", "i should", "let me", "i will",
                "the user wants", "the user is asking"
            ]
            if any(phrase in thought.lower() for phrase in generic_phrases):
                warnings.append("Thought contains generic phrasing")
        
        elif item.output.status == "complete":
            # Check 5: Final Answer Quality
            answer = item.output.final_answer.strip()
            if len(answer) < cls.MIN_FINAL_ANSWER_LENGTH:
                return ValidationResult(
                    is_valid=False,
                    error_type="quality",
                    error_message=f"Final answer too short: {len(answer)} chars"
                )
            
            # Check for vague answers
            vague_phrases = ["i don't know", "not sure", "maybe", "perhaps", "i think"]
            if any(phrase in answer.lower() for phrase in vague_phrases):
                warnings.append("Final answer contains vague language")
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    @classmethod
    def validate_domain_logic(cls, item: TrainingExample) -> ValidationResult:
        """
        Layer 3: Domain-specific validation (tool arguments sanity checks).
        This ensures tools are used correctly.
        """
        if item.output.status != "running":
            return ValidationResult(is_valid=True)
        
        tool = item.output.tool_use
        tool_name = tool.tool_name
        args = tool.arguments
        
        # Tool-specific validation
        if tool_name == "codebase_search":
            query = args.query.strip()
            if len(query) < 2:
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message="Codebase search query too short"
                )
            
            # Check for overly generic searches
            if query.lower() in ["code", "file", "function", "class", "todo"]:
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message=f"Search query too generic: '{query}'"
                )
        
        elif tool_name == "file_manager":
            # Path validation
            if not args.path or args.path.strip() == "":
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message="File manager missing path"
                )
            
            # Operation-specific checks
            if args.operation == "write":
                if not args.content:
                    return ValidationResult(
                        is_valid=False,
                        error_type="domain",
                        error_message="Write operation missing content"
                    )
            
            elif args.operation == "patch":
                # target_string must not be None or empty
                if not args.target_string:  
                    return ValidationResult(
                        is_valid=False,
                        error_type="domain",
                        error_message="Patch operation missing target_string"
                    )
                # replacement_string must not be None; empty "" is VALID
                if args.replacement_string is None:
                    return ValidationResult(
                        is_valid=False,
                        error_type="domain",
                        error_message="Patch operation missing replacement_string (empty string allowed)"
                    )
                
                # Ensure target and replacement are different
                if args.target_string == args.replacement_string:
                    return ValidationResult(
                        is_valid=False,
                        error_type="domain",
                        error_message="Patch target and replacement are identical"
                    )
        
        elif tool_name == "sandbox_exec":
            code = args.code.strip()
            if not code:
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message="Sandbox execution missing code"
                )
            
            # Check for dangerous patterns (even in synthetic data)
            dangerous_patterns = ["rm -rf", "os.system", "__import__", "eval("]
            if any(pattern in code for pattern in dangerous_patterns):
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message="Sandbox code contains dangerous patterns"
                )
        
        elif tool_name == "ask_human":
            question = args.question.strip()
            if len(question) < 5:
                return ValidationResult(
                    is_valid=False,
                    error_type="domain",
                    error_message="ask_human question too short"
                )
            
            # List of dangerous keywords in user_query
            dangerous_keywords = ["delete", "drop", "truncate", "format", "shutdown", "kill"]
            # Ensure it's actually a question; if it's not a typical question, allow it if the user_query contains a dangerous command
            if not any(word in question.lower() for word in ["?", "what", "how", "which", "should", "can", "could"]):
                if not any(word in item.user_query.lower() for word in dangerous_keywords):
                    return ValidationResult(
                        is_valid=False,
                        error_type="domain",
                        error_message="ask_human content doesn't appear to be a question"
                    )
        
        return ValidationResult(is_valid=True)
    
    @classmethod
    def validate_full(cls, item: TrainingExample) -> ValidationResult:
        """
        Run all validation layers in sequence.
        Short-circuits on first failure for efficiency.
        """
        # Layer 1: Structure
        result = cls.validate_structural(item)
        if not result.is_valid:
            return result
        
        # Layer 2: Quality
        result = cls.validate_quality(item)
        if not result.is_valid:
            return result
        
        # Layer 3: Domain Logic
        result = cls.validate_domain_logic(item)
        return result
    
    @staticmethod
    def _is_parroting(query: str, thought: str, threshold: float = 0.8) -> bool:
        """
        Detect if thought is just parroting the query.
        Uses simple character overlap for efficiency.
        """
        # Normalize
        q_norm = query.lower().strip()[:50]
        t_norm = thought.lower().strip()[:50]
        
        # Check exact prefix match
        if t_norm.startswith(q_norm[:20]):
            return True
        
        # Check word overlap (Jaccard similarity)
        q_words = set(q_norm.split())
        t_words = set(t_norm.split())
        
        if not q_words or not t_words:
            return False
        
        intersection = len(q_words & t_words)
        union = len(q_words | t_words)
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold

# --- Batch Processing Utilities ---

def validate_batch(items: List[TrainingExample], 
                   strict: bool = True,
                   log_errors: bool = True) -> Tuple[List[TrainingExample], Dict[str, int]]:
    """
    Validate a batch of training examples.
    
    Args:
        items: List of TrainingExample objects
        strict: If False, return invalid items with warnings
        log_errors: Whether to log validation failures
    
    Returns:
        Tuple of (valid_items, stats_dict)
    """
    validator = DataValidator()
    valid_items = []
    stats = {
        "total": len(items),
        "valid": 0,
        "invalid_structural": 0,
        "invalid_quality": 0,
        "invalid_domain": 0,
        "warnings": 0
    }
    
    for i, item in enumerate(items):
        result = validator.validate_full(item)
        
        if result.is_valid:
            valid_items.append(item)
            stats["valid"] += 1
            
            if result.warnings:
                stats["warnings"] += len(result.warnings)
                if log_errors:
                    logger.warning(f"Item {i}: {', '.join(result.warnings)}")
        else:
            if result.error_type:
                stats[f"invalid_{result.error_type}"] += 1
            
            if log_errors:
                print(f"❌ DROP REASON: {result.error_type} - {result.error_message}")
                print(f"   Query: {item.user_query[:50]}...")
                logger.error(f"Item {i} failed ({result.error_type}): {result.error_message}")
    
    return valid_items, stats

def validate_jsonl_file(filepath: Path, 
                        max_items: Optional[int] = None) -> Dict[str, int]:
    """
    Validate an entire JSONL file and return statistics.
    Useful for auditing existing datasets.
    """
    
    validator = DataValidator()
    stats = {
        "total": 0,
        "valid": 0,
        "invalid_structural": 0,
        "invalid_quality": 0,
        "invalid_domain": 0,
        "parse_errors": 0
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            
            line = line.strip()
            if not line:
                continue
            
            stats["total"] += 1
            
            try:
                data = json.loads(line)
                item = TrainingExample(**data)
                result = validator.validate_full(item)
                
                if result.is_valid:
                    stats["valid"] += 1
                else:
                    stats[f"invalid_{result.error_type}"] += 1
                    if result.error_type == "domain":
                        snippet = json.dumps(data, ensure_ascii=False)[:120]
                        logger.error(f"Line {i+1} domain error: {result.error_message}\n  snippet: {snippet}...")
            
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                logger.error(f"Line {i+1}: JSON decode error")
            except Exception as e:
                stats["parse_errors"] += 1
                logger.error(f"Line {i+1}: {str(e)}")
    
    return stats