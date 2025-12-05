from textwrap import dedent
from src.config import INTENT_DISTRIBUTION, DOMAINS, PERSONAS, QUERY_STYLES
import random

def build_generation_prompt(intent_config, domain, persona, query_style, batch_size):
    tool_name = intent_config['tool']

    tool_logic = _get_tool_logic(tool_name)
    schema_constraints = _get_schema_constraints(tool_name)
    style_guide = _get_style_guide(query_style, 2)
    examples = _get_few_shot_examples(tool_name)

    prompt = dedent(f"""
    # ROLE: Elite Synthetic Data Generator for AI Coding Agents
    
    You are generating training data for a Router model that maps user queries → structured tool calls.
    
    ## TASK
    Generate {batch_size} diverse, realistic training examples.
    
    ## CONTEXT
    - **Domain:** {domain}
    - **Persona:** {persona}
    - **Target Tool:** {tool_name if tool_name else "NONE (Direct Answer)"}
    - **Intent:** {intent_config['desc']}
    - **Query Style:** {query_style}

    ## CRITICAL: LOGIC & CONSISTENCY RULES
    {tool_logic}

    ## STYLE GUIDELINES
    {style_guide}

    ## THOUGHT QUALITY (CRITICAL)
    **MINIMUM 6 WORDS OR THE PIPELINE CRASHES** 
    
    Your thought must explain:
    1. **What** you're doing
    2. **Why** you chose this tool
    3. **How** it helps solve the request
    
    BAD (Too short):
    - "Search for it." (3 words - REJECTED)
    - "Check the config file." (4 words - REJECTED)
    - "Run code to test." (4 words - REJECTED)
    
    GOOD (Detailed reasoning):
    - "User wants auth logic, so I'll search semantically across the codebase to find relevant implementations." (16 words)
    - "I need to verify this regex pattern handles edge cases correctly, so I'll execute it in the sandbox with test inputs." (21 words)
    - "The user specified the exact file path, so I'll use file_manager to read the contents directly." (17 words)
    
    **Formula:** "I need to [ACTION] because [REASON], so I'll [TOOL] to [OUTCOME]."

    ## OUTPUT SCHEMA CONSTRAINTS
    You must output JSON that strictly adheres to this structure.
    {schema_constraints}

    ## EXAMPLES (Reference these patterns)
    {examples}
    
    ## FINAL NEGATIVE CONSTRAINTS
    1. DO NOT generate 'null' values. If a field is optional and not needed, omit it completely.
    2. DO NOT output Markdown blocks (\`\`\`json). Just the raw lines of JSON objects.
    3. Ensure every JSON object is on a single line.
    
    GENERATE {batch_size} EXAMPLES NOW:
    """)
    
    return prompt

def _get_tool_logic(tool_name):
    """Defines when and why each tool should be selected."""

    if tool_name == "codebase_search":
        return """
        **CODEBASE SEARCH: Discovery & Exploration**
        
        **When to use:**
        - User doesn't know the file path
        - Looking for patterns/concepts
        - Exploring unfamiliar code
        - Finding examples of usage
        
        **Mode selection guide:**
        - **exact:** Specific symbols (e.g., "class User", "function authenticate", "CONFIG_KEY")
        - **semantic:** Concepts (e.g., "how does auth work", "payment processing logic")
        - **hybrid:** Mixed (e.g., "database connection setup", "API error handling")
        
        **File pattern usage:**
        - Include ONLY if user mentions scope: "in tests", "backend folder", "*.config files"
        - Otherwise OMIT the field entirely (don't search all files explicitly)
        
        **Anti-patterns (NEVER generate these):**
        - User query contains full path: "Read src/auth.py" → This is file_manager
        - User query asks to modify: "Update the timeout" → This is file_manager
        - User query has code to run: "Test this snippet" → This is sandbox_exec
        """
        
    elif tool_name == "file_manager":
        return """
        **FILE MANAGER: Direct File Operations**
        
        **When to use:**
        - User provides explicit file path (must be IN the query)
        - Operations: list directory, read file, write file, patch file
        
        **CRITICAL RULE:**
        The user query MUST contain the file path or directory.
        You must invent realistic paths and embed them naturally.
        
        **Good user queries:**
        - "Read the config in src/settings.json"
        - "Update line 45 in backend/auth/handler.py"
        - "List all files in tests/unit/"
        - "Write this schema to models/user.py"
        
        **Operation-specific args:**
        - list: only needs `path`
        - read: only needs `path`
        - write: needs `path` + `content`
        - patch: needs `path` + `target_string` + `replacement_string`
        
        **Anti-patterns:**
        - Vague queries: "Find the config" → This is search
        - No path: "Update the timeout" → Need file path
        """
        
    elif tool_name == "sandbox_exec":
        return """
        **SANDBOX EXEC: Code Execution & Validation**
        
        **When to use:**
        - Test/validate code snippets
        - Run calculations or data processing
        - Reproduce bugs with minimal example
        - Verify algorithm behavior
        - Prototype quick solutions
        
        **Code requirements:**
        - Must be valid, runnable Python
        - Use print() for output
        - Keep it focused (< 50 lines ideal)
        
        **User query patterns:**
        - "Test if this regex works: ..."
        - "Run this calculation: ..."
        - "Check if list comprehension is faster"
        - "Validate this JSON parser"
        
        **Anti-patterns:**
        - Query asks to modify codebase → file_manager
        - Query asks to search for code → codebase_search
        """
        
    elif tool_name == "ask_human":
        return """
        **ASK HUMAN: Escalation & Clarification**
        
        **When to use:**
        1. **Dangerous operations:** Delete DB, drop tables, rm -rf
        2. **Ambiguous requests:** "Update the server" (which one?)
        3. **Business logic needed:** "Refund this user" (what's the policy?)
        4. **Permission required:** "Deploy to production"
        5. **Insufficient context:** "Fix the bug" (which bug?)
        
        **CRITICAL DISTINCTION:**
        - User query = The TRIGGER (vague/dangerous request)
        - Agent output = The QUESTION (asking for clarification)
        
        BAD user query: "I need permission to proceed"
        GOOD user query: "Delete all logs older than 30 days"
        GOOD agent response: "This affects 10GB of data. Proceed with deletion?"
        
        **Question quality:**
        - Be specific about what you need
        - Explain why you're asking
        - Provide context in the `context` field
        """
        
    else:  # Direct answer
        return """
        **DIRECT ANSWER: Explanation & Guidance**
        
        **When to use:**
        - Conceptual questions (no code execution needed)
        - Explanations of errors/concepts
        - Best practice advice
        - "How do I..." questions
        - Casual conversation
        
        **User query patterns:**
        - "How does async/await work?"
        - "What's the difference between X and Y?"
        - "Why am I getting this error?"
        - "What are best practices for..."
        
        **Anti-patterns:**
        - Queries that need code inspection → search
        - Queries that need file changes → file_manager
        - Queries that need code testing → sandbox
        """
    
    return ""

def _get_schema_constraints(tool_name):
    """Shows exact JSON structure with validation rules."""
    
    common_header = """
    **JSON Structure (one object per line):**
    """
    
    if tool_name == "codebase_search":
        return common_header + """
    {
      "user_query": "String matching persona style and query_style",
      "output": {
        "status": "running",
        "thought": "Detailed reasoning (minimum 6 words, target 15+ words)",
        "tool_use": {
          "tool_name": "codebase_search",
          "arguments": {
            "query": "Search term (identifier or concept)",
            "mode": "exact" | "semantic" | "hybrid",
            "file_pattern": "glob pattern (OPTIONAL - omit if searching everywhere)"
          }
        }
      }
    }
    
    **Validation:**
    - `mode` must match query type (exact for symbols, semantic for concepts)
    - `file_pattern` only if user mentions scope
    - `thought` must explain mode choice
    """
    
    elif tool_name == "file_manager":
        return common_header + """
    {
      "user_query": "String with explicit file path mentioned",
      "output": {
        "status": "running",
        "thought": "Detailed reasoning (minimum 6 words)",
        "tool_use": {
          "tool_name": "file_manager",
          "arguments": {
            "operation": "list" | "read" | "write" | "patch",
            "path": "Relative file/directory path",
            "content": "String (ONLY if operation=write)",
            "target_string": "String (ONLY if operation=patch)",
            "replacement_string": "String (ONLY if operation=patch)"
          }
        }
      }
    }
    
    **Validation:**
    - `path` must match path mentioned in user_query
    - `content` required for write, forbidden for others
    - `target_string` + `replacement_string` required for patch, forbidden for others
    - Omit unused fields (don't set to null)
    """
    
    elif tool_name == "sandbox_exec":
        return common_header + """
    {
      "user_query": "String requesting code execution/validation",
      "output": {
        "status": "running",
        "thought": "Detailed reasoning (minimum 6 words)",
        "tool_use": {
          "tool_name": "sandbox_exec",
          "arguments": {
            "code": "Valid Python code string",
            "timeout": 30
          }
        }
      }
    }
    
    **Validation:**
    - `code` must be syntactically valid Python
    - Use print() for outputs
    - `timeout` defaults to 30 (only change if query mentions time constraint)
    """
    
    elif tool_name == "ask_human":
        return common_header + """
    {
      "user_query": "Vague/dangerous/ambiguous request",
      "output": {
        "status": "running",
        "thought": "Explain why escalation is needed (minimum 6 words)",
        "tool_use": {
          "tool_name": "ask_human",
          "arguments": {
            "question": "Specific question for clarification",
            "context": "Why you need this info (OPTIONAL)"
          }
        }
      }
    }
    
    **Validation:**
    - `question` must be specific and actionable
    - `context` explains the situation (optional but recommended)
    - User query != question (user triggers, agent asks)
    """
    
    else:  # Direct answer
        return common_header + """
    {
      "user_query": "Conceptual/explanatory question",
      "output": {
        "status": "complete",
        "final_answer": "Comprehensive response (2-5 sentences)"
      }
    }
    
    **Validation:**
    - No `thought` or `tool_use` fields
    - `final_answer` should be helpful and complete
    - Match persona's technical level
    """

def _get_style_guide(selected_style: str = None, num_examples: int = 3) -> str:
    """
    Format query style information for prompt injection.
    
    Args:
        selected_style: If provided, only show this style. If None, show one random style.
        num_examples: Number of examples to show per style (default: 3)
    
    Returns:
        Formatted style guide string ready for prompt injection
    """
    
    # If no style provided, randomly select one
    if selected_style is None:
        selected_style = random.choice(list(QUERY_STYLES.keys()))
    
    # Validate style exists
    if selected_style not in QUERY_STYLES:
        raise ValueError(f"Unknown style: {selected_style}. Must be one of {list(QUERY_STYLES.keys())}")
    
    style_info = QUERY_STYLES[selected_style]
    
    # Build formatted output
    guide = dedent(f"""
    **Selected Style: `{selected_style}`**
    
    **Description:** {style_info['desc']}
    
    **Examples of this style:**
    """).strip()
    
    # Add examples (limit to num_examples)
    examples = style_info['examples'][:num_examples]
    for i, example in enumerate(examples, 1):
        guide += f"\n    {i}. \"{example}\""
    
    # Add application instruction
    guide += dedent("""
    
    **CRITICAL:** Every 'user_query' you generate MUST exhibit this communication style.
    The style should feel natural and authentic to the persona, not forced or artificial.
    """).strip()
    
    return guide

def _get_few_shot_examples(tool_name):
    """Provides concrete examples to anchor the LLM's output."""
    
    if tool_name == "codebase_search":
        return """
    **Example 1 (exact mode):**
    {"user_query": "Where is the User class defined?", "output": {"status": "running", "thought": "User is looking for a specific class definition, so I'll use exact mode to find the symbol precisely in the codebase.", "tool_use": {"tool_name": "codebase_search", "arguments": {"query": "class User", "mode": "exact"}}}}
    
    **Example 2 (semantic mode):**
    {"user_query": "How do we handle authentication failures?", "output": {"status": "running", "thought": "This is a conceptual question about auth error handling, so I'll use semantic search to find relevant logic across the codebase.", "tool_use": {"tool_name": "codebase_search", "arguments": {"query": "authentication failure handling", "mode": "semantic"}}}}
    
    **Example 3 (with file_pattern):**
    {"user_query": "Find all test cases for the payment module", "output": {"status": "running", "thought": "User specifically mentioned tests, so I'll search with a file pattern to scope the results to test directories only.", "tool_use": {"tool_name": "codebase_search", "arguments": {"query": "payment test cases", "mode": "hybrid", "file_pattern": "tests/**/*.py"}}}}
    """
    
    elif tool_name == "file_manager":
        return """
    **Example 1 (read):**
    {"user_query": "Show me what's in src/config.py", "output": {"status": "running", "thought": "User specified the exact file path, so I'll use file_manager to read the contents of src/config.py directly.", "tool_use": {"tool_name": "file_manager", "arguments": {"operation": "read", "path": "src/config.py"}}}}
    
    **Example 2 (patch):**
    {"user_query": "Change the timeout from 30 to 60 in backend/api/client.py", "output": {"status": "running", "thought": "User wants to modify a specific value in a known file, so I'll use patch operation to replace the timeout value.", "tool_use": {"tool_name": "file_manager", "arguments": {"operation": "patch", "path": "backend/api/client.py", "target_string": "timeout = 30", "replacement_string": "timeout = 60"}}}}
    """
    
    elif tool_name == "sandbox_exec":
        return """
    **Example 1:**
    {"user_query": "Test if this regex works: ^[a-z]+@[a-z]+\\\\..+$", "output": {"status": "running", "thought": "User wants to validate a regex pattern, so I'll execute it in the sandbox with test cases to verify it works correctly.", "tool_use": {"tool_name": "sandbox_exec", "arguments": {"code": "import re\\npattern = r'^[a-z]+@[a-z]+\\\\..+$'\\ntest_cases = ['test@example.com', 'invalid', 'Test@example.com']\\nfor case in test_cases:\\n    print(f'{case}: {bool(re.match(pattern, case))}')", "timeout": 30}}}}
    """
    
    elif tool_name == "ask_human":
        return """
    **Example 1 (dangerous action):**
    {"user_query": "Drop the users table from the database", "output": {"status": "running", "thought": "This is a destructive database operation that could cause data loss, so I need explicit human confirmation before proceeding.", "tool_use": {"tool_name": "ask_human", "arguments": {"question": "This will permanently delete all user data. Are you sure you want to drop the users table?", "context": "Dropping tables is irreversible and may affect production systems."}}}}
    
    **Example 2 (ambiguous):**
    {"user_query": "Deploy the new feature", "output": {"status": "running", "thought": "The request is ambiguous because there are multiple deployment environments, so I need clarification on which environment to target.", "tool_use": {"tool_name": "ask_human", "arguments": {"question": "Which environment should I deploy to: staging, production, or development?", "context": "Need to know target environment before proceeding with deployment."}}}}
    """
    
    else:  # Direct answer
        return """
    **Example 1:**
    {"user_query": "What's the difference between async and await in Python?", "output": {"status": "complete", "final_answer": "In Python, 'async' defines a coroutine function that can be paused and resumed, while 'await' is used inside async functions to pause execution until an awaitable object (like another coroutine) completes. Together, they enable asynchronous programming for I/O-bound operations."}}
    """


# Example usage
if __name__ == "__main__":
    # 1. Weighted Random Selection of Intent
    intent_config = random.choices(
        INTENT_DISTRIBUTION, 
        weights=[x['weight'] for x in INTENT_DISTRIBUTION],
        k=1
    )[0]
    
    # 2. Randomize context
    domain = random.choice(DOMAINS)
    persona = random.choice(PERSONAS)
    prompt = build_generation_prompt(
        intent_config=intent_config,
        domain=domain,
        persona=persona,
        query_style=random.choice(list(QUERY_STYLES.keys())),
        batch_size=5
    )
    
    print(prompt)