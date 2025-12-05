import time
import random
from src.client import get_client
from src.schemas import BatchResponse
from src.prompts import build_generation_prompt
from src.config import GROQ_MODELS, FALLBACK_MODEL, QUERY_STYLES
from src.logger import logger

def generate_batch(intent_config, domain, persona, batch_size=5, retry_count=0):
    """
    Generates a batch of examples using model roulette and exponential backoff.
    """
    query_style = random.choice(list(QUERY_STYLES.keys()))
    prompt = build_generation_prompt(intent_config, domain, persona, query_style, batch_size)
    
    # 1. Select Model
    if retry_count > 0:
        model_id = FALLBACK_MODEL
        logger.warning(f"  Retry #{retry_count}: Switching to Fallback ({model_id})")
    else:
        model_id = random.choice(GROQ_MODELS)
    
    instructor_client = get_client(model_id)

    try:
        resp = instructor_client.chat.completions.create(
            model=model_id,
            response_model=BatchResponse,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85, # High diversity  
            max_retries=2,
        )
        return resp.items

    except Exception as e:
        error_msg = str(e)
        
        # RATE LIMIT HANDLING
        if "429" in error_msg or "ResourceExhausted" in error_msg or "Too Many Requests" in error_msg:
            wait_time = (2 ** retry_count) + random.uniform(1, 5)
            logger.warning(f"   Rate limit on {model_id.split('/')[-1]}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            if retry_count < 4:
                return generate_batch(intent_config, domain, persona, batch_size, retry_count + 1)
        
        # Validation handling
        elif "validation" in error_msg:
            logger.warning(f"   Validation failed on {model_id.split('/')[-1]}. Retrying...")
            if retry_count < 3:
                return generate_batch(intent_config, domain, persona, batch_size, retry_count + 1)
        
        logger.error(f"Critical Error: {e}")
        return []