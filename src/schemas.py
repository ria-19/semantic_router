# Semantic Router - QLoRA fine-tuned Llama-3.1-8B-Instruct
# Author: Riya Sangwan
# License: MIT
# Repository: https://github.com/ria-19/semantic-router


from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from typing import Literal, Optional, Union, List

# --- Shared Config ---
# This tells Pydantic: "If OpenAI/Google adds extra fields like 'type' or 'ref', ignore them."
base_config = ConfigDict(extra="ignore")

# ==========================================
# ARGUMENT SCHEMAS (The Payloads)
# ==========================================

class CodebaseSearchArguments(BaseModel):
    model_config = base_config
    query: str = Field(
        ...,
        description="The search term (e.g., 'class User' or 'def auth').")
    mode: Literal["exact", "semantic", "hybrid"] = Field(
        default="hybrid",
        description="'exact' for symbols, 'semantic' for concepts."
    )
    file_pattern: Optional[str] = Field(None, description="Glob pattern (e.g. 'src/*.py'). Leave NULL to search all files.")
    

class FileManagerArguments(BaseModel):
    model_config = base_config
    operation: Literal["list", "read", "write", "patch"] = Field(..., description="Action.")
    path: str = Field(..., description="Relative path (e.g., 'src/utils.py').")
    content: Optional[str] = Field(None, description="Required for 'write'.")
    target_string: Optional[str] = Field(None, description="Required for 'patch'.")
    replacement_string: Optional[str] = Field(None, description="Required for 'patch'.")

    @model_validator(mode='after')
    def check_args(self):
        if self.operation == "write" and not self.content:
            raise ValueError("Operation 'write' requires 'content'.")
        if self.operation == "patch" and (not self.target_string or not self.replacement_string):
            raise ValueError("Operation 'patch' requires 'target' and 'replacement'.")
        return self

class SandboxExecArguments(BaseModel):
    model_config = base_config
    code: str = Field(
        ..., 
        description="Python code to run. Use print() to see output."
    )
    timeout: int = Field(30, description="Max execution time in seconds.")

class AskHumanArguments(BaseModel):
    model_config = base_config
    question: str = Field(..., description="Question for the user.")
    context: Optional[str] = Field(None, description="Why this is needed.")

# ==========================================
# TOOL DEFINITIONS (The wrappers)
# =======================================

class CodebaseSearchTool(BaseModel):
    model_config = base_config
    tool_name: Literal["codebase_search"]
    arguments: CodebaseSearchArguments

class FileManagerTool(BaseModel):
    model_config = base_config
    tool_name: Literal["file_manager"]
    arguments: FileManagerArguments

class SandboxExecTool(BaseModel):
    model_config = base_config
    tool_name: Literal["sandbox_exec"]
    arguments: SandboxExecArguments

class AskHumanTool(BaseModel):
    model_config = base_config
    tool_name: Literal["ask_human"]
    arguments: AskHumanArguments

# ==========================================
# STATE DEFINITIONS (Discriminated Union)
# ==========================================

# STATE A: RUNNING (Thinking & Acting)
# class AgentRunning(BaseModel):  
#     model_config = base_config
#     status: Literal["running"] = "running"
    
#     # Chain of Thought: Crucial for smart agents
#     thought: str = Field(..., description="Internal reasoning: strictly 1-2 lines. Explain why this tool was chosen.")    
#     # The Discriminated Union:
#     # The LLM MUST pick exactly one valid tool structure.
#     tool_use: Union[
#         CodebaseSearchTool, 
#         FileManagerTool, 
#         SandboxExecTool, 
#         AskHumanTool
#     ] = Field(..., discriminator="tool_name")

#     @field_validator('thought')
#     def validate_thought(cls, v):
#         if '\n' in v:
#             raise ValueError("thought must be single line")
#         word_count = len(v.split())
#         if word_count < 8:
#             raise ValueError(f"thought must be at least 8 words (got {word_count})")
#         if word_count > 50:
#             raise ValueError(f"thought must be under 50 words (got {word_count})")
#         return v.strip()

# # STATE B: COMPLETE (Answering)
# class AgentFinish(BaseModel):   
#     model_config = base_config
#     status: Literal["complete"] = "complete"
#     final_answer: str = Field(..., description="The final response to the user.")

# # ==========================================
# # 4. MASTER UNION
# # ==========================================

# AgentOutput = Union[AgentRunning, AgentFinish]

# # ==========================================
# # 5. TRAINING DATA WRAPPERS (For Generator)
# # ==========================================

# class TrainingExample(BaseModel):
#     model_config = base_config
#     user_query: str = Field(..., description="The simulated user request.")
#     output: AgentOutput = Field(..., description="The expected agent response.")

# class BatchResponse(BaseModel):
#     model_config = base_config
#     items: List[TrainingExample] = Field(..., description="A batch of valid training examples.")


# ==========================================
# 3. FLATTENED AGENT OUTPUT (The Fix)
# ==========================================

class AgentOutput(BaseModel):
    """
    A single unified class that handles both 'running' and 'complete' states.
    This prevents 'anyOf' schema errors in Groq.
    """
    model_config = base_config
    
    status: Literal["running", "complete"] = Field(..., description="Agent status.")
    
    # Optional fields (logic enforced by validator below)
    thought: Optional[str] = Field(None, description="Reasoning (required if running).")
    
    tool_use: Optional[Union[
        CodebaseSearchTool, 
        FileManagerTool, 
        SandboxExecTool, 
        AskHumanTool
    ]] = Field(None, discriminator="tool_name")
    
    final_answer: Optional[str] = Field(None, description="Response (required if complete).")

    @model_validator(mode='after')
    def validate_structure(self):
        # Clean up thought string
        if self.thought:
             self.thought = " ".join(self.thought.split()).strip()

        if self.status == "running":
            if not self.tool_use:
                raise ValueError("Status 'running' requires 'tool_use'.")
            if not self.thought:
                raise ValueError("Status 'running' requires 'thought'.")
            
            # Word count check
            word_count = len(self.thought.split())
            if word_count < 6:
                raise ValueError(f"Thought too short ({word_count} words). Must be > 6 words.")
            
            # Ensure final_answer is empty
            self.final_answer = None

        elif self.status == "complete":
            if not self.final_answer:
                raise ValueError("Status 'complete' requires 'final_answer'.")
            
            # Ensure tool fields are empty
            self.tool_use = None
            self.thought = None  # Or keep it if you want CoT for answers too
            
        return self

# ==========================================
# 4. TRAINING DATA WRAPPERS
# ==========================================

class TrainingExample(BaseModel):
    model_config = base_config
    user_query: str
    output: AgentOutput

class BatchResponse(BaseModel):
    model_config = base_config
    items: List[TrainingExample]