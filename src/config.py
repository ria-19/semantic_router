"""
Configuration settings for the Synthetic Data Generator.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ConfigDict

# 1. LOAD ENV VARS
load_dotenv()

# 2. PATHS
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS & MODELS
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY and not GOOGLE_API_KEY:
    print("WARNING: No API Keys found in .env (GROQ_API_KEY or GOOGLE_API_KEY)")
    
# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Dec 2025 "Model Roulette" to avoid Rate Limits
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # The Teacher (High IQ, strict limits)
    # "mixtral-8x7b-32768",       # The Workhorse (Reliable)
    "llama-3.1-8b-instant",     # The Speedster (High Throughput)
    # "gemma2-9b-it"              # The Wildcard (Diversity)
]

# Fallback model if others fail specifically
FALLBACK_MODEL = "llama-3.1-8b-instant"

# =============================================================================
# GENERATION SETTINGS
# =============================================================================

DOMAINS = [
    "E-Commerce API", "Video Game Engine", "Crypto Trading Bot",
    "Machine Learning Pipeline", "Legacy Banking System", "Healthcare EMR",
    "IoT Fleet Management", "Real-Time Analytics", "Embedded Robotics",
    "Social Media Recommender", "Cloud Cost Optimizer", "Autonomous Vehicle Sim",
    "SaaS Billing System", "Telecom Provisioning", "Customer Support Chatbot",
    "Supply Chain Logistics Platform", "Cybersecurity SIEM", "Content Management System",
    "Ad Tech Bidding Engine", "Genomics Processing Pipeline", "Smart Grid Controller",
    "Drone Delivery System", "Legal Document Automation", "Property Management System",
    "Restaurant POS System", "Airline Reservation System", "Weather Forecasting Service",
    "EdTech Learning Platform", "Streaming Media CDN", "Agricultural Sensor Network",
    "Municipal Traffic Control", "Insurance Claims Processing", "Energy Trading Platform",
    "Clinical Trial Management", "Subscription Box Fulfillment", "Fraud Detection System",
    "Virtual Event Platform", "Construction Project Tracker", "Fleet Telematics API"
]

PERSONAS = [
    # Original personas
    "Junior Intern (Vague, nervous, uses simple language)",
    "Senior Engineer (Technical, precise, mentions specific patterns/frameworks)",
    "Product Manager (Functional description, no code details, business-focused)",
    "Security Auditor (Paranoid, looking for vulnerabilities, mentions CVEs)",
    "DevOps Engineer (Focused on config and deployment, mentions infrastructure)",
    "QA Tester (Edge cases, reproduction steps, bug reports with stack traces)",
    "SRE (Latency concerns, dashboards, SLIs/SLOs, outages)",
    "Data Scientist (Stats-heavy, unclear about infra, mentions metrics)",
    "Research Engineer (Experimental, messy requests, academic terminology)",
    "Tech Lead (High-level, architectural decisions, mentions trade-offs)",
    "Customer Support Agent (Non-technical, user complaints, emotional language)",
    "DBA (Schema changes, migrations, indexing, query optimization)",
    "Compliance Officer (Risk, audit trail, logs, regulatory requirements)",
    "Performance Engineer (Profiling, optimization, bottlenecks, memory leaks)",
    "Founder/CTO (Visionary but vague, mentions competitors and market)",
    "Solutions Architect (Multi-system integrations, vendor evaluation, big picture)",
    "Mobile Developer (Platform-specific issues, native APIs, app lifecycle concerns)",
    "Frontend Engineer (UI/UX focused, browser compatibility, state management)",
    "Backend Engineer (API contracts, database design, service boundaries)",
    "Infrastructure Engineer (Networking, bare metal, capacity planning, hardware)",
    "Security Engineer (Threat modeling, cryptography, authentication flows)",
    "ML Engineer (Model serving, feature stores, training pipelines, GPU utilization)",
    "Release Manager (Deployment windows, rollback plans, versioning strategy)",
    "Business Analyst (KPIs, reporting requirements, data definitions, workflows)",
    "UX Researcher (User behavior, analytics interpretation, A/B test results)",
    "Technical Writer (Documentation gaps, API reference accuracy, examples)",
    "Consultant (Cross-company context, best practices, industry standards)",
    "Contractor (Limited context, time-boxed tasks, handoff concerns)",
    "Open Source Contributor (External perspective, generalization, community standards)",
    "Executive Stakeholder (Budget implications, timeline concerns, strategic alignment)",
    "Night Shift Operator (Incident response, limited resources, production issues)",
    "Integration Partner (Third-party API, contract negotiations, SLA discussions)",
    "Academic Advisor (Theoretical approach, literature references, novel algorithms)",
    "Regulatory Inspector (Certification requirements, evidence collection, standards)",
    "Startup Technical Co-founder (Scrappy, hacky solutions, rapid iteration)"
]

# THE GOLDEN DISTRIBUTION 
INTENT_DISTRIBUTION = [
    # TOOLS (85%)
    {
        "intent": "search",
        "tool": "codebase_search",
        "weight": 0.35,
        "desc": "Look up functions, trace where logic lives, inspect configs, verify how a feature is wired, explore unfamiliar modules, find examples of patterns, locate error messages, discover API usage, identify dependencies, or map data flows."
    },
    {
        "intent": "compute",
        "tool": "sandbox_exec",
        "weight": 0.24,
        "desc": "Evaluate code snippets, validate algorithms, run quick calculations, test input/output behavior, experiment with API request flows, reproduce bugs, verify regex patterns, benchmark performance, simulate edge cases, or prototype solutions."
    },
    {
        "intent": "modify",
        "tool": "file_manager",
        "weight": 0.18,
        "desc": "Apply small fixes, adjust configuration values, update constants, clean up formatting, introduce safe incremental refactors, add logging, update dependencies, rename variables, remove dead code, or implement minor features."
    },
    {
        "intent": "escalate",
        "tool": "ask_human",
        "weight": 0.08,
        "desc": "Requests blocked by missing context, unclear business rules, security-sensitive actions, operations requiring human approval, ambiguous requirements, conflicting priorities, architectural decisions, policy questions, or tasks outside automation scope."
    },
    # DIRECT ANSWERS (15%)
    {
        "intent": "answer",
        "tool": None,
        "weight": 0.15,
        "desc": "Provide explanations, clarify concepts, respond to casual questions, handle requests that can't be executed programmatically, explain error messages, suggest approaches, discuss trade-offs, offer best practices, give historical context, or provide education."
    }
]

QUERY_STYLES = {
    # ==========================================
    # BASIC COMMUNICATION STYLES
    # ==========================================
    "direct": {
        "desc": "Straightforward command or statement",
        "examples": ["Find the auth handler", "Show me the config", "Search for User class"]
    },
    "question": {
        "desc": "Phrased as an interrogative",
        "examples": ["Where is the auth handler?", "How does caching work?", "What's in the config file?"]
    },
    "problem": {
        "desc": "Describes an issue or blocker",
        "examples": ["I can't find the auth handler", "The tests are failing", "Getting errors in production"]
    },
    "context": {
        "desc": "Provides background before the request",
        "examples": ["We're refactoring auth, need to find the handler", "For the API redesign, show me current routes"]
    },
    "urgent": {
        "desc": "Time-sensitive with pressure indicators",
        "examples": ["ASAP: find auth handler", "URGENT: production is down", "Need this NOW for deploy"]
    },
    "confused": {
        "desc": "Uncertain, seeking guidance",
        "examples": ["Not sure where auth stuff is...", "I think it's in utils?", "Maybe check the handlers?"]
    },
    "comparative": {
        "desc": "Contrasts options or approaches",
        "examples": ["Should I use redis or memcache?", "Which is better: JWT or sessions?"]
    },
    "exploratory": {
        "desc": "Open-ended investigation",
        "examples": ["Let's see how auth works", "Curious about the payment flow", "Exploring the database layer"]
    },
    
    # ==========================================
    # EMOTIONAL/TONAL STYLES
    # ==========================================
    "imperative": {
        "desc": "Direct command, authoritative tone",
        "examples": ["Do this now", "Fix the bug immediately", "Deploy to production"]
    },
    "passive_aggressive": {
        "desc": "Indirect frustration or sarcasm",
        "examples": ["I guess we'll just ignore the timeout issue", "Not sure why this was never documented"]
    },
    "overly_polite": {
        "desc": "Excessive courtesy, apologetic",
        "examples": ["Sorry to bother, but could you possibly find...", "If it's not too much trouble..."]
    },
    "emotional": {
        "desc": "Frustrated, excited, or stressed tone",
        "examples": ["This is driving me crazy!", "Finally found the issue!", "I'm so stressed about this"]
    },
    "sarcastic": {
        "desc": "Dry humor or ironic phrasing",
        "examples": ["Oh great, another memory leak", "Sure, let's just break production", "Perfect, more tech debt"]
    },
    "delegating": {
        "desc": "Assigning task to someone/something else",
        "examples": ["Can you handle this?", "Please take care of the deployment", "You figure out the auth"]
    },
    
    # ==========================================
    # STRUCTURAL STYLES
    # ==========================================
    "hypothetical": {
        "desc": "Poses scenarios or possibilities",
        "examples": ["What if we moved auth to a microservice?", "Suppose we used GraphQL instead..."]
    },
    "narrative": {
        "desc": "Long story with embedded request",
        "examples": ["So I was debugging yesterday and noticed the cache wasn't invalidating, which led me to wonder about..."]
    },
    "fragmented": {
        "desc": "Incomplete thoughts, stream of consciousness",
        "examples": ["auth handler... somewhere in backend? maybe utils...", "need to find... timeout config... settings?"]
    },
    "acronym_heavy": {
        "desc": "Filled with jargon and abbreviations",
        "examples": ["Need the JWT impl ASAP for SSO integration", "FYI the API CRUD ops need ORM refactor"]
    },
    "code_mixed": {
        "desc": "Natural language mixed with code snippets",
        "examples": ["Find where we call authenticate() with the user param", "Search for imports of redis.Redis"]
    },
    "multi_part": {
        "desc": "Several questions/requests in one message",
        "examples": ["Find auth handler, also check timeout config, and see if tests exist"]
    },
    "follow_up": {
        "desc": "Assumes previous context",
        "examples": ["Also check the utils", "And another thing about that config...", "Oh and the timeout too"]
    },
    "specification": {
        "desc": "Formal requirements-style language",
        "examples": ["The system shall implement authentication via OAuth2", "Requirements: search capability with semantic indexing"]
    },
    "diagnostic": {
        "desc": "Describes troubleshooting steps taken",
        "examples": ["I tried searching in backend/, then checked utils/, saw it imported from auth/"]
    },
    "minimal": {
        "desc": "One or two words, ultra terse",
        "examples": ["auth handler", "config?", "timeout", "tests"]
    },
    "verbose": {
        "desc": "Overly detailed, 10x longer than needed",
        "examples": ["I would like to respectfully request that you perform a comprehensive search of the entire codebase..."]
    },
    "rubber_duck": {
        "desc": "Talking through problem, may self-answer",
        "examples": ["So the auth isn't working... probably the handler... let me find where it's defined first..."]
    },
    "checklist": {
        "desc": "Bulleted or numbered list of items",
        "examples": ["1. Find auth handler 2. Check config 3. Run tests", "TODO: search for timeout, update value, verify"]
    },
    "screenshot_dependent": {
        "desc": "References external context not provided",
        "examples": ["See attached error log", "As shown in the diagram", "Like in the screenshot"]
    },
    "time_constrained": {
        "desc": "Mentions deadlines or time pressure",
        "examples": ["Need this before EOD", "Deploy by 5pm", "Have a meeting in 1hr, need this ready"]
    },
    "cross_functional": {
        "desc": "Mentions multiple teams or systems",
        "examples": ["Backend team needs frontend to check API", "Sync with DevOps on deployment config"]
    },
    "philosophical": {
        "desc": "Questions fundamentals or design decisions",
        "examples": ["Why do we even use JWT?", "Should we rethink our auth strategy?", "Is REST the right choice?"]
    },
    
    # ==========================================
    # ROUTING-SPECIFIC STYLES (Tool Selection Hints)
    # ==========================================
    "implicit_search": {
        "desc": "Implies search without saying 'search'",
        "examples": ["Where is the auth handler", "How does caching work here", "Show me the payment logic"]
    },
    "implicit_execute": {
        "desc": "Implies code execution without explicit request",
        "examples": ["Try running this snippet", "What happens if I call auth() with null", "Test this regex"]
    },
    "implicit_modify": {
        "desc": "Assumes permission to edit files",
        "examples": ["Change timeout to 60", "Update the config", "Fix the typo in line 45"]
    },
    "implicit_escalate": {
        "desc": "Blocked tone, needs help without asking",
        "examples": ["Not sure what to do here", "This seems risky", "I don't have enough context"]
    },
    "ambiguous_intent": {
        "desc": "Could be search OR answer, unclear",
        "examples": ["Tell me about authentication", "Config settings", "How does this work"]
    },
    "chained_request": {
        "desc": "Multiple sequential operations",
        "examples": ["Find the handler, then update timeout, then run tests", "Search, modify, validate"]
    },
    "conditional_logic": {
        "desc": "If/else logic in natural language",
        "examples": ["If config exists update it, otherwise create new", "Check tests, if passing deploy"]
    },
    "negative_phrasing": {
        "desc": "Uses negative constructions",
        "examples": ["Don't see why this fails", "Can't find the config", "Isn't working as expected"]
    },
    "assumption_laden": {
        "desc": "Assumes context that may not exist",
        "examples": ["Update the usual config", "Fix it like last time", "Use the standard approach"]
    },
    "tool_agnostic": {
        "desc": "Describes outcome, not method",
        "examples": ["I need to know the timeout value", "Want to see test results", "Need auth working"]
    },
    "meta_request": {
        "desc": "Asking how to do something",
        "examples": ["Help me figure out how to auth", "What's the best way to cache", "Guide me through deployment"]
    },
    "partial_path": {
        "desc": "Incomplete file/function reference",
        "examples": ["Something in utils about auth", "Handler in backend somewhere", "Config file, not sure where"]
    },
    "error_dump": {
        "desc": "Pastes stack trace or error log",
        "examples": ["Getting: TypeError: cannot read property 'user' of undefined at auth.js:45", "[Full stack trace...]"]
    },
    "permission_uncertain": {
        "desc": "Asks if action is allowed",
        "examples": ["Can I delete this?", "Should I modify production config?", "Am I allowed to drop this table?"]
    },
    "scope_creep": {
        "desc": "Request expands mid-message",
        "examples": ["Find auth handler, oh and also while you're at it check if tests exist and update docs"]
    },
    "false_precision": {
        "desc": "Sounds specific but lacks key details",
        "examples": ["Update line 45", "Change the config", "Fix the handler (which one?)"]
    },
    "tool_name_dropped": {
        "desc": "Mentions tool incorrectly",
        "examples": ["Grep the API for auth", "Git blame the config", "SSH into the handler"]
    },
    "read_only_intent": {
        "desc": "Only wants information, no changes",
        "examples": ["Show me the config", "What's in this file", "Display current settings"]
    },
    "mutation_intent": {
        "desc": "Clearly wants to modify something",
        "examples": ["Update the timeout", "Fix this bug", "Delete old logs", "Refactor the handler"]
    },
    "validation_request": {
        "desc": "Wants to verify/test something",
        "examples": ["Check if this regex works", "Validate the JSON schema", "Test the API endpoint"]
    },
    "documentation_query": {
        "desc": "Asks for explanation or guidance",
        "examples": ["How do I use JWT?", "What's the auth flow?", "Explain async/await"]
    },
    "precedent_seeking": {
        "desc": "References past implementations",
        "examples": ["How did we handle auth in v1?", "Find the old payment logic", "What was the previous approach"]
    },
    "configuration_query": {
        "desc": "Asks about settings/values",
        "examples": ["What's the current timeout?", "Show me database config", "What port are we using?"]
    },
    "ownership_question": {
        "desc": "Asks who/what is responsible",
        "examples": ["Who owns the auth service?", "What team maintains this?", "Which module handles payments?"]
    },
    "impact_analysis": {
        "desc": "Asks about consequences",
        "examples": ["What breaks if I change this?", "What depends on this function?", "Impact of removing cache?"]
    },
    "compatibility_check": {
        "desc": "Verifies if things work together",
        "examples": ["Does this work with Python 3.9?", "Is JWT compatible with our setup?", "Will this break IE11?"]
    },
    "optimization_vague": {
        "desc": "Requests improvement without specifics",
        "examples": ["Make it faster", "Optimize this", "Improve performance", "Speed up the queries"]
    },
    "security_paranoid": {
        "desc": "Security-focused concerns",
        "examples": ["Is this safe?", "Any security issues?", "Could this be exploited?", "Check for vulnerabilities"]
    },
    "regex_embedded": {
        "desc": "Contains regex pattern",
        "examples": ["Find files matching *.py", "Search for pattern ^auth_.*$", "Grep for /api/v[0-9]+/"]
    },
    "json_payload": {
        "desc": "Includes JSON or code block",
        "examples": ["Run this: {\"user\": \"test\"}", "Execute: def foo(): return 42", "Test this payload"]
    },
    "file_path_present": {
        "desc": "Explicitly mentions file path",
        "examples": ["Read src/auth.py", "Update config/settings.json", "Check backend/handlers/user.py"]
    },
    "function_signature": {
        "desc": "Includes function call syntax",
        "examples": ["Find authenticate(user, password)", "Search for getUserById(id)", "Where is init()?"]
    },
    "env_var_reference": {
        "desc": "Mentions environment variables",
        "examples": ["What's DATABASE_URL set to?", "Check API_KEY config", "Show me ENV vars"]
    },
    "dependency_question": {
        "desc": "Asks about relationships",
        "examples": ["What uses this function?", "What depends on auth module?", "Which files import this?"]
    },
    "historical_context": {
        "desc": "Asks why something was done",
        "examples": ["Why was this added?", "History of this change?", "Who introduced this pattern?"]
    },
    "testing_scenario": {
        "desc": "Describes test case",
        "examples": ["Does this handle null input?", "Test edge case: empty string", "Verify it works with unicode"]
    },
    "migration_inquiry": {
        "desc": "Asks about transitioning systems",
        "examples": ["How do we migrate from Redis to Memcache?", "Moving from REST to GraphQL", "Upgrade path?"]
    },
    "integration_question": {
        "desc": "Asks how components connect",
        "examples": ["How does frontend call backend?", "Integration between auth and DB?", "API flow diagram?"]
    },
    "rollback_concern": {
        "desc": "Asks about undoing changes",
        "examples": ["How do I undo this?", "Rollback the deployment", "Revert to previous config"]
    }
}

# --- THE SINGLE SOURCE OF TRUTH ---
SYSTEM_PROMPT = """You are the Semantic Brain of an autonomous AI engineer.
Your role is to route user queries to the correct tool or answer directly.

OUTPUT RULES:
1. If the user asks a question you can answer with general knowledge, return status="complete".
2. If the user asks for a specific action (search, file edit, debug), return status="running" and choose the tool.
3. If the request is ambiguous or impossible, return status="running" and use the 'ask_human' tool.
4. Output STRICT JSON only. No markdown, no yapping."""

# --- Shared Config ---
# This tells Pydantic: "If OpenAI/Google adds extra fields like 'type' or 'ref', ignore them."
BASE_CONFIG = ConfigDict(extra="ignore")

# --- VALIDATION CONFIGURATION ---
VALIDATION_CONFIG = {
    # Quality Checks
    "MIN_QUERY_LENGTH": int(os.getenv("MIN_QUERY_LENGTH", 5)),
    "MIN_THOUGHT_WORDS": int(os.getenv("MIN_THOUGHT_WORDS", 8)),
    "MAX_THOUGHT_WORDS": int(os.getenv("MAX_THOUGHT_WORDS", 100)),
    "MIN_FINAL_ANSWER_LENGTH": int(os.getenv("MIN_FINAL_ANSWER_LENGTH", 10)),
    "PARROTING_THRESHOLD": float(os.getenv("PARROTING_THRESHOLD", 0.8)),
    
    # Domain Checks
    "MIN_SEARCH_QUERY_LENGTH": int(os.getenv("MIN_SEARCH_QUERY_LENGTH", 2)),
    "MAX_FILE_SIZE_KB": int(os.getenv("MAX_FILE_SIZE_KB", 512)),
    
    # Batching and Retry
    "GENERATION_BATCH_SIZE": int(os.getenv("GENERATION_BATCH_SIZE", 5)),
    "MAX_GENERATION_RETRIES": int(os.getenv("MAX_GENERATION_RETRIES", 4)),
}

TOTAL_TARGET = 20


# ----------------------------------------------------
# CONFIGURATION FOR HUGGINGFACE
# ----------------------------------------------------

HF_USERNAME = "tai-tai-sama"


