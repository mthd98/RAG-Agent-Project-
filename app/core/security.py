from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from core.config import settings
from fastapi import Security, HTTPException, status, Request # Import Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials # Import HTTPBearer related classes
from core.config import settings
import logging # Import logging

logger = logging.getLogger(__name__) # Set up logger for this module

# Define the custom X-API-Key header (optional now, but good to keep for direct API calls)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False) # Set auto_error=False

# Define the standard Authorization: Bearer header scheme
bearer_scheme = HTTPBearer(auto_error=False) # Set auto_error=False

async def get_api_key(
    request: Request, # Inject the request object
    api_key_header_value: str = Security(api_key_header),
    bearer_token: HTTPAuthorizationCredentials = Security(bearer_scheme)
):
    """
    Dependency to validate the API key.
    Checks for EITHER a valid X-API-Key header OR a valid Authorization: Bearer token.
    Compares the found key/token with the one in our config.
    """
    received_key = None

    # 1. Check for X-API-Key header first
    if api_key_header_value:
        logger.debug("Found X-API-Key header.")
        received_key = api_key_header_value

    # 2. If no X-API-Key, check for Authorization: Bearer token
    elif bearer_token:
        logger.debug("Found Authorization: Bearer header.")
        received_key = bearer_token.credentials # Extract the token part

    # 3. Validate the received key (if any) against settings
    if received_key and received_key == settings.API_KEY:
        logger.info("API Key validation successful.")
        return received_key # Return the valid key
    else:
        # Log the failed attempt, being careful not to log the actual key if present
        if received_key:
             logger.warning("API Key validation failed: Invalid key received.")
        else:
             logger.warning("API Key validation failed: No key found in X-API-Key or Authorization header.")

        # Raise the exception if validation fails or no key was found
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"}, # Standard for Bearer auth failure
        )
