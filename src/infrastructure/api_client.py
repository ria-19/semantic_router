import instructor
from openai import AsyncOpenAI
from ..config import GROQ_API_KEY, GOOGLE_API_KEY
from .logger import logger


def get_client(provider: str = "groq", async_mode: bool = True):
    """
    Returns a provider-agnostic Async client wrapped by Instructor.
    """
    
    # 1. Configuration Map
    # The 'openai/v1' suffix is critical for these compatibility endpoints
    CONFIG = {
        "groq": {
            "api_key": GROQ_API_KEY,
            "base_url": "https://api.groq.com/openai/v1",
            "mode": instructor.Mode.JSON,
        },
        "google": {
            "api_key": GOOGLE_API_KEY,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", 
            "mode": instructor.Mode.JSON,
        }
    }

    if provider not in CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {list(CONFIG.keys())}")

    settings = CONFIG[provider]
    
    if not settings["api_key"]:
        logger.error(f"API Key for {provider} is missing.")
        raise ValueError(f"{provider.upper()}_API_KEY not found in env.")

    try:
        # 2. Universal Async Client
        base_client = AsyncOpenAI(
            base_url=settings["base_url"],
            api_key=settings["api_key"],
        )

        # 3. Patch
        client = instructor.patch(
            base_client, 
            mode=settings["mode"]
        )
        
        return client

    except Exception as e:
        logger.error(f"Failed to initialize {provider} client: {e}")
        raise e