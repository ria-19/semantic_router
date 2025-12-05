import os
from dotenv import load_dotenv
import instructor
from instructor import from_provider, Mode

def get_client(model_id: str):
    """
    Returns an instructor client configured for the specific model.
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY. Please set it in environment variables or .env file.")

    model = 'groq/' + model_id

    return from_provider(
        model=model,  
        api_key=api_key,
        mode=Mode.GENAI_STRUCTURED_OUTPUTS
    )


# Initialize singleton
# instructor_client = get_client()



