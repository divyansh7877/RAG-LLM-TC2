"""
Keycloak OpenID Connect (OIDC) based authentication system.
"""
import logging
from typing import Optional, Dict, Any, List
from authlib.integrations.starlette_client import OAuth
from authlib.jose import jwt, JsonWebKey
from joserfc.errors import ExpiredTokenError, InvalidTokenError
from .config import config
import json
import urllib.request

# Set up logging
logger = logging.getLogger(__name__)

class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass

class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired."""
    pass

class TokenInvalidError(AuthenticationError):
    """Raised when JWT token is invalid."""
    pass

class AuthenticationManager:
    """
    Manages authentication against a Keycloak server using OIDC.

    This class handles the validation of JWT tokens issued by Keycloak.
    """
    def __init__(self):
        self.keycloak_server_url = config.KEYCLOAK_SERVER_URL
        self.keycloak_realm = config.KEYCLOAK_REALM
        self.keycloak_client_id = config.KEYCLOAK_CLIENT_ID
        self.algorithm = config.KEYCLOAK_ALGORITHM

        self.metadata_url = f"{self.keycloak_server_url}realms/{self.keycloak_realm}/.well-known/openid-configuration"
        self.jwk_set = None

    async def load_jwks(self):
        """Loads and parses the JSON Web Key Set (JWKS) from Keycloak."""
        try:
            oauth = OAuth()
            oauth.register(
                name='keycloak',
                server_metadata_url=self.metadata_url,
                client_kwargs={'scope': 'openid email profile'},
            )
            server_metadata = await oauth.keycloak.load_server_metadata()
            jwks_uri = server_metadata['jwks_uri']

            # Fetch JWKS JSON and import as a key set
            with urllib.request.urlopen(jwks_uri, timeout=5) as resp:
                jwks_json = json.load(resp)
            self.jwk_set = JsonWebKey.import_key_set(jwks_json)
            logger.info("Successfully loaded and parsed JWKS from Keycloak.")
        except Exception as e:
            logger.error(f"Failed to load JWKS from Keycloak: {e}")
            raise AuthenticationError("Could not connect to Keycloak to get public keys.")

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decodes and validates a JWT token from Keycloak.

        Args:
            token: The JWT token string.

        Returns:
            The decoded token payload as a dictionary.

        Raises:
            TokenExpiredError: If the token has expired.
            TokenInvalidError: If the token is invalid for any other reason.
        """
        if not self.jwk_set:
            raise AuthenticationError("JWKS not loaded. Cannot validate token.")

        try:
            # Decode the token using the public keys from Keycloak's JWKS
            claims = jwt.decode(
                s=token,
                key=self.jwk_set,
                claims_options={
                    'iss': {'essential': True, 'value': f"{self.keycloak_server_url}realms/{self.keycloak_realm}"},
                    # Allow audience to be validated manually to support str or list
                    'aud': {'essential': True},
                }
            )
            # Manual audience validation (supports str or list) with Keycloak compatibility:
            # Accept if 'aud' includes our client OR 'azp' equals our client (typical Keycloak token has aud='account').
            aud = claims.get('aud')
            azp = claims.get('azp')
            if isinstance(aud, str):
                valid_audience = (
                    aud == self.keycloak_client_id or
                    (aud == 'account' and azp == self.keycloak_client_id)
                )
            else:
                aud_list = aud or []
                valid_audience = (
                    self.keycloak_client_id in aud_list or
                    ('account' in aud_list and azp == self.keycloak_client_id)
                )
            if not valid_audience:
                raise InvalidTokenError("Invalid audience")
            claims.validate()
            return claims
        except ExpiredTokenError:
            raise TokenExpiredError("Token has expired.")
        except InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {e}")
        except Exception as e:
            raise TokenInvalidError(f"An unexpected error occurred during token validation: {e}")

# Global authentication manager instance
auth_manager = AuthenticationManager()
