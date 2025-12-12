import time
import random
import asyncio
from typing import List, Dict, Any

from pydantic import ValidationError
from openai import RateLimitError, APIError

from src.infrastructure import get_client, logger
from src.schemas import BatchResponse
from .prompt_builder import build_generation_prompt
from src.config import GROQ_MODELS, FALLBACK_MODEL, QUERY_STYLES

async def generate_batch(intent_config, domain, persona, batch_size=5, retry_count=0) -> List[Dict[str, Any]]:
    """
    Generates a batch of examples using model roulette and exponential backoff.
    
    Args:
        intent_config: Configuration dict for the current training intent.
        domain: The domain of the query (e.g., 'File I/O').
        persona: The persona of the user (e.g., 'Architect').
        batch_size: Number of examples to request.
        retry_count: Current retry level for exponential backoff.
        
    Returns:
        A list of validated dictionaries ready for saving.
    """

    query_style = random.choice(list(QUERY_STYLES.keys()))
    prompt = build_generation_prompt(intent_config, domain, persona, query_style, batch_size)
    
    # 1. Select Model
    if retry_count > 0:
        model_id = FALLBACK_MODEL
        logger.warning(f"  Retry #{retry_count}: Switching to Fallback ({model_id})")
    else:
        model_id = random.choice(GROQ_MODELS)
    
    instructor_client = get_client()

    try:
        resp = await instructor_client.chat.completions.create(
            model=model_id,
            response_model=BatchResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85, # High diversity  
            max_retries=2,
        )

        logger.info(f"   Success with {model_id.split('/')[-1]}. Generated {len(resp.items)} items.")
        return resp.items

    except RateLimitError as e:
        wait_time = (2 ** retry_count) + random.uniform(1, 3)
        logger.warning(f"   Rate Limit Hit. Waiting {wait_time:.2f}s.")
        await asyncio.sleep(wait_time)
        
        if retry_count < 4:
            return await generate_batch(intent_config, domain, persona, batch_size, retry_count + 1)

    except ValidationError as e:
        # Schema validation error
        logger.warning(f"   Schema Validation failed. Retrying (Attempt {retry_count + 1}).")
        if retry_count < 3:
            return await generate_batch(intent_config, domain, persona, batch_size, retry_count + 1)

    except APIError as e:
        # General API error (e.g., 500 server error, invalid key, etc.)
        logger.error(f"   API Error ({e.status_code}) on {model_id.split('/')[-1]}: {e.message}")
        # Do not retry general API errors immediately; they are often not transient
        pass 

    except Exception as e:
        # Catch any remaining unexpected errors (e.g., network timeout, unexpected Python error)
        logger.exception(f"   Critical Unknown Error: {e}") 

    return [] # Final failure case