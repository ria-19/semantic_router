import json
import os

def save_batch(batch_items, output_file):
    """
    Validates and saves a batch of TrainingExample items to a JSONL file.
    
    Features:
    - Removes 'None' fields (Null Tax).
    - Validates 'thought' quality (length and uniqueness).
    - Enforces tool-specific logic constraints.
    """
    valid_count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for item in batch_items:
            try:
                # ---------------------------------------------------------
                # 1. STRUCTURAL & QUALITY VALIDATION
                # ---------------------------------------------------------
                
                # Check 1: User Query existence
                if not item.user_query or len(item.user_query.strip()) < 5:
                    print(f"Skipping: Query too short")
                    continue

                if item.output.status == "running":
                    # Check 2: Thought Quality
                    thought = item.output.thought
                    thought_words = len(thought.split())
                    
                    # Length Check (Strict 8-50 words)
                    if thought_words < 8:
                        print(f"Skipping: Thought too short ({thought_words} words)")
                        continue
                    if thought_words > 50:
                        print(f"Skipping: Thought too long ({thought_words} words)")
                        continue

                    # Parroting Check (Did it just copy the query?)
                    query_start = item.user_query.lower()[:20]
                    thought_start = thought.lower()[:20]
                    if query_start == thought_start:
                        print(f"Skipping: Thought parallels query too closely (Parroting)")
                        continue

                    # Check 3: Tool Specific Sanity Checks
                    tool = item.output.tool_use
                    args = tool.arguments
                    
                    if tool.tool_name == "codebase_search":
                        # Ensure we aren't searching for empty strings
                        if len(args.query.strip()) < 2:
                            continue
                            
                    elif tool.tool_name == "file_manager":
                        # Redundant check (Pydantic handles this, but good for safety)
                        if args.operation == "write" and not args.content:
                            print(f"Skipping: Write operation missing content")
                            continue
                        if args.operation == "patch" and (not args.target_string or not args.replacement_string):
                            print(f"Skipping: Patch operation missing details")
                            continue

                    elif tool.tool_name == "sandbox_exec":
                        # Ensure code isn't empty
                        if not args.code.strip():
                            continue

                # ---------------------------------------------------------
                # 2. SERIALIZATION (NO NULL TAX)
                # ---------------------------------------------------------
                
                # CRITICAL: exclude_none=True removes fields like 
                # "content": null, "target_string": null, etc.
                json_str = item.model_dump_json(exclude_none=True)
                
                f.write(json_str + "\n")
                valid_count += 1

            except Exception as e:
                print(f"Validation/Save Error: {e}")
                continue
                
    return valid_count