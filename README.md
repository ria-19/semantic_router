# 🧠 The Semantic Router (Fine-Tuned Agent Brain)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Unsloth](https://img.shields.io/badge/Training-Unsloth-green)
![Llama 3](https://img.shields.io/badge/Model-Llama_3.1-blueviolet)
![Pydantic](https://img.shields.io/badge/Validation-Pydantic-e92063)

> **Phase 1: Synthetic Data Engineering (Complete)**  
> **Phase 2: QLoRA Fine-Tuning (Ready)**

## 🎯 Mission
To replace massive, high-latency System Prompts in Agent architectures with a **specialized, fine-tuned Adapter**. 

This project builds the "Brain" of an Autonomous Coding Agent. Instead of relying on a generic LLM to guess how to use tools, we are fine-tuning **Llama-3.1-8B** to function as a deterministic **Intent Router** that converts natural language queries into strict, executable JSON tool calls with < 200ms latency.

## 🏗️ Architecture

We moved beyond simple scripting to a modular **Data Factory** approach. The pipeline generates high-diversity training examples (Domains + Personas + Edge Cases) while enforcing strict logic constraints.

### The File Structure
```text
.
├── generate_data.py       # 🚀 Main entry point for the data factory
├── src/
│   ├── schemas.py         # 📜 The Contract: Pydantic definitions for all tools
│   ├── prompts.py         # 🧠 The Brain: Advanced prompt logic (CoT, Anti-Hallucination)
│   ├── generator.py       # 🏭 The Factory: Multi-model generation logic (Groq/Google)
│   ├── client.py          # 🔌 The Connection: Instructor client setup
│   ├── formatting.py      # 🎨 ChatML Formatter: JSON -> Llama-3 Training Format
│   └── utils.py           # 🛡️ The Guard: Validation & Null-Tax removal
├── data/
│   └── raw/               # 💾 JSONL files containing (User -> Thought -> Tool)
└── notebooks/             # 📓 Colab notebooks for Unsloth training
```

## 💎 The "Golden" Data Strategy
This is not just random synthetic data. The pipeline implements **Logic-Aware Generation** to prevent common agent failures:

1.  **🚫 Anti-Hallucination (The "Magic Path" Fix)**
    *   *Constraint:* The generator distinguishes between **Discovery** (Search) and **Action** (File Ops).
    *   *Result:* The agent never attempts to read a file (`file_manager`) unless the user explicitly provides the path or context.

2.  **📉 Avoiding the "Null Tax"**
    *   *Constraint:* Output schemas use `exclude_none=True`.
    *   *Result:* We save ~20% token usage per example by stripping empty fields (e.g., `content: null` during read operations), resulting in faster inference.

3.  **💭 Structured Chain-of-Thought (CoT)**
    *   *Constraint:* Enforced minimum word counts and reasoning patterns.
    *   *Result:* The model doesn't just parrot the user; it explains *why* it chose a tool (e.g., *"User is asking for a concept, so I must use semantic search, not exact match"*).

4.  **🎰 "Model Roulette" Generation**
    *   *Strategy:* We rotate between **Llama-3.3-70B** (Logic), **Mixtral-8x7B** (Stability), and **Gemma-2-9B** (Diversity) to prevent "Model Collapse" and avoid API rate limits.

## 🛠️ The Toolset (Schema)
The fine-tuned model is trained to route requests to these four deterministic tools:

| Tool | Capability | Logic Constraint |
| :--- | :--- | :--- |
| **`codebase_search`** | RAG / Semantic Search | Must choose `exact` vs `semantic` mode based on query type. |
| **`file_manager`** | Read / Write / Patch | Requires explicit paths. No guessing. |
| **`sandbox_exec`** | Python Interpreter | For calculation, verification, or logic testing only. |
| **`ask_human`** | Human-in-the-Loop | Triggered by ambiguity or high-risk actions (e.g., DB deletion). |

## 🚀 Usage

This project uses `uv` for modern, fast Python dependency management.

### 1. Setup Environment
Create a `.env` file with your API keys (for generation):
```bash
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
HF_TOKEN=hf_...
```

### 2. Generate Synthetic Data
Run the factory to create `data/raw/router_train.jsonl`.
```bash
uv run generate_data.py
```
*Note: Check `src/config.py` to adjust batch size, domains, and personas.*

### 3. Format for Llama-3
Convert the raw JSONL into the ChatML format required by Unsloth.
```bash
uv run src/formatting.py
```

### 4. Upload to HuggingFace
Version control your dataset before training.
```bash
uv run src/upload_data.py
```

## 📚 Knowledge Base
Check the `docs/` folder for my research notes on:
*   **LoRA / PEFT**: Why we freeze base weights.
*   **Unsloth**: Optimization techniques for 2x faster training.
*   **Data Hygiene**: Schema validation rules.

