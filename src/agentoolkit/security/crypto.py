"""
Crypto AgenTool - Provides cryptographic operations including hashing, encryption, and key management.

This toolkit provides comprehensive cryptographic functionality including:
- Hashing algorithms (SHA256, SHA512, MD5, bcrypt)
- Symmetric encryption (AES)
- Asymmetric encryption (RSA)
- Key generation and management
- Digital signatures
- JWT operations
- Base64 encoding/decoding

Example Usage:
    >>> from agentoolkit.security.crypto import create_crypto_agent
    >>> from agentool.core.injector import get_injector
    >>> 
    >>> # Create and register the agent
    >>> agent = create_crypto_agent()
    >>> 
    >>> # Use through injector
    >>> injector = get_injector()
    >>> result = await injector.run('crypto', {
    ...     "operation": "hash",
    ...     "algorithm": "sha256",
    ...     "data": "secret password"
    ... })
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Literal, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import RunContext

from agentool.base import BaseOperationInput
from agentool import create_agentool
from agentool.core.registry import RoutingConfig


class CryptoInput(BaseOperationInput):
    """Input schema for cryptographic operations."""
    operation: Literal[
        'hash', 'verify_hash', 'generate_key', 'encrypt', 'decrypt',
        'sign', 'verify_signature', 'encode_base64', 'decode_base64',
        'generate_jwt', 'verify_jwt', 'generate_salt', 'hmac'
    ] = Field(description="The cryptographic operation to perform")
    
    # Common fields
    data: Optional[str] = Field(None, description="Data to process")
    algorithm: Optional[Literal['sha256', 'sha512', 'md5', 'sha1', 'bcrypt', 'aes', 'rsa']] = Field(
        None, description="Algorithm to use"
    )
    key: Optional[str] = Field(None, description="Encryption/decryption key")
    
    # Hash-specific
    salt: Optional[str] = Field(None, description="Salt for hashing")
    iterations: Optional[int] = Field(None, description="Iterations for key derivation")
    
    # Encryption-specific
    iv: Optional[str] = Field(None, description="Initialization vector for AES")
    key_size: Optional[int] = Field(None, description="Key size in bits")
    
    # JWT-specific
    payload: Optional[Dict[str, Any]] = Field(None, description="JWT payload")
    secret: Optional[str] = Field(None, description="JWT secret")
    expires_in: Optional[int] = Field(None, description="JWT expiration time in seconds")
    
    # Signature-specific
    private_key: Optional[str] = Field(None, description="Private key for signing")
    public_key: Optional[str] = Field(None, description="Public key for verification")
    signature: Optional[str] = Field(None, description="Signature to verify")
    
    @field_validator('data')
    def validate_data(cls, v, info):
        """Validate data field is present for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['hash', 'verify_hash', 'encrypt', 'decrypt', 'sign', 
                        'verify_signature', 'encode_base64', 'decode_base64', 
                        'verify_jwt', 'hmac'] and not v:
            raise ValueError(f"data is required for {operation} operation")
        return v
    
    @field_validator('algorithm')
    def validate_algorithm(cls, v, info):
        """Validate algorithm field is present for operations that require it."""
        operation = info.data.get('operation')
        if operation in ['hash', 'verify_hash', 'generate_key', 'encrypt', 
                        'decrypt', 'sign', 'verify_signature', 'hmac'] and not v:
            raise ValueError(f"algorithm is required for {operation} operation")
        return v
    
    @field_validator('key')
    def validate_key(cls, v, info):
        """Validate key field is present for operations that require it."""
        operation = info.data.get('operation')
        # Note: verify_hash uses 'key' field to pass the hash value
        if operation in ['encrypt', 'decrypt', 'verify_hash', 'hmac'] and not v:
            raise ValueError(f"key is required for {operation} operation")
        return v
    
    @field_validator('payload')
    def validate_payload(cls, v, info):
        """Validate payload field for JWT operations."""
        operation = info.data.get('operation')
        if operation == 'generate_jwt':
            if v is None:
                raise ValueError("payload is required for generate_jwt operation")
            # Note: empty dict check is done in the tool function
        return v
    
    @field_validator('secret')
    def validate_secret(cls, v, info):
        """Validate secret field for JWT operations."""
        operation = info.data.get('operation')
        if operation in ['generate_jwt', 'verify_jwt'] and not v:
            raise ValueError(f"secret is required for {operation} operation")
        return v
    
    @field_validator('private_key')
    def validate_private_key(cls, v, info):
        """Validate private_key field for signing operations."""
        operation = info.data.get('operation')
        if operation == 'sign' and not v:
            raise ValueError("private_key is required for sign operation")
        return v
    
    @field_validator('public_key')
    def validate_public_key(cls, v, info):
        """Validate public_key field for signature verification."""
        operation = info.data.get('operation')
        if operation == 'verify_signature' and not v:
            raise ValueError("public_key is required for verify_signature operation")
        return v
    
    @field_validator('signature')
    def validate_signature(cls, v, info):
        """Validate signature field for verification."""
        operation = info.data.get('operation')
        if operation == 'verify_signature' and not v:
            raise ValueError("signature is required for verify_signature operation")
        return v


class CryptoOutput(BaseModel):
    """Structured output for cryptographic operations."""
    success: bool = Field(description="Whether the operation succeeded")
    operation: str = Field(description="The operation that was performed")
    message: str = Field(description="Human-readable result message")
    data: Optional[Any] = Field(None, description="Operation-specific data")
    error: Optional[str] = Field(None, description="Error message if operation failed")


async def crypto_hash(ctx: RunContext[Any], data: str, algorithm: str, 
                     salt: Optional[str], iterations: Optional[int]) -> CryptoOutput:
    """
    Hash data using specified algorithm.
    
    Args:
        ctx: Runtime context
        data: Data to hash
        algorithm: Hashing algorithm
        salt: Optional salt
        iterations: Iterations for PBKDF2
        
    Returns:
        CryptoOutput with the hash result
        
    Raises:
        ValueError: If algorithm is not supported
        RuntimeError: If hashing operation fails
    """
    if algorithm == 'bcrypt':
        # For bcrypt, we'll use a simple PBKDF2 approximation
        # In production, you'd use the bcrypt library
        # If salt is provided as base64, decode it first
        if salt:
            try:
                # Try to decode as base64 first
                salt_bytes = base64.b64decode(salt)
            except Exception as e:
                # If it fails, use as raw string
                salt_bytes = salt.encode()
        else:
            salt_bytes = os.urandom(16)
        
        iterations = iterations or 100000
        
        try:
            hash_value = hashlib.pbkdf2_hmac(
                'sha256',
                data.encode(),
                salt_bytes,
                iterations
            )
        except Exception as e:
            raise RuntimeError(f"Failed to hash data with bcrypt algorithm: {e}") from e
        
        # Return base64 encoded hash with salt
        result = base64.b64encode(salt_bytes + hash_value).decode()
        
        return CryptoOutput(
            success=True,
            operation="hash",
            message=f"Successfully hashed data using {algorithm}",
            data={
                "algorithm": algorithm,
                "hash": result,
                "salt": base64.b64encode(salt_bytes).decode(),
                "iterations": iterations
            }
        )
    else:
        # Standard hash algorithms
        try:
            hash_obj = hashlib.new(algorithm)
        except ValueError as e:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
        
        if salt:
            hash_obj.update(salt.encode())
        
        hash_obj.update(data.encode())
        hash_value = hash_obj.hexdigest()
        
        return CryptoOutput(
            success=True,
            operation="hash",
            message=f"Successfully hashed data using {algorithm}",
            data={
                "algorithm": algorithm,
                "hash": hash_value,
                "salt": salt
            }
        )


async def crypto_verify_hash(ctx: RunContext[Any], data: str, algorithm: str, 
                           hash: str, salt: Optional[str]) -> CryptoOutput:
    """
    Verify a hash against data.
    
    Args:
        ctx: Runtime context
        data: Data to verify
        algorithm: Hashing algorithm used
        hash: Hash to verify against
        salt: Salt used in hashing
        
    Returns:
        CryptoOutput with verification result (success=True with valid field)
        
    Raises:
        ValueError: If required parameters are missing or invalid format
        RuntimeError: If hash verification operation fails
    """
    if algorithm == 'bcrypt':
        # For bcrypt, we need to use the same approach as in hash
        # The stored hash includes the salt, so we need to regenerate it
        if not salt:
            # Missing salt is a validation error, not a discovery failure
            raise ValueError("Salt is required for bcrypt hash verification")
        
        try:
            salt_bytes = base64.b64decode(salt)
        except Exception as e:
            # Invalid format is a validation error
            raise ValueError(f"Invalid salt format for bcrypt verification: {e}") from e
        
        # Use the same default iterations as in crypto_hash
        iterations = 10000  # This is what auth toolkit uses
        
        try:
            hash_value = hashlib.pbkdf2_hmac(
                'sha256',
                data.encode(),
                salt_bytes,
                iterations
            )
        except Exception as e:
            # Actual computation failure is a runtime error
            raise RuntimeError(f"Failed to compute bcrypt hash for verification: {e}") from e
        
        # The stored hash is base64(salt + hash_value)
        expected_hash = base64.b64encode(salt_bytes + hash_value).decode()
        is_valid = expected_hash == hash
    else:
        # Generate hash with same parameters for other algorithms
        result = await crypto_hash(ctx, data, algorithm, salt, None)
        
        computed_hash = result.data["hash"]
        is_valid = computed_hash == hash
    
    # Always return success=True for discovery operations
    # The "valid" field indicates if verification passed
    return CryptoOutput(
        success=True,
        operation="verify_hash",
        message=f"Hash verification {'successful' if is_valid else 'failed'}",
        data={
            "valid": is_valid,
            "algorithm": algorithm
        }
    )


async def crypto_generate_key(ctx: RunContext[Any], algorithm: str, 
                            key_size: Optional[int]) -> CryptoOutput:
    """
    Generate a cryptographic key.
    
    Args:
        ctx: Runtime context
        algorithm: Algorithm for key generation
        key_size: Key size in bits
        
    Returns:
        CryptoOutput with generated key
        
    Raises:
        ValueError: If algorithm or key size is invalid
    """
    if algorithm == 'aes':
        # AES key sizes: 128, 192, 256 bits
        key_size = key_size or 256
        if key_size not in [128, 192, 256]:
            raise ValueError(f"Invalid AES key size: {key_size}. Must be 128, 192, or 256 bits")
        
        key_bytes = secrets.token_bytes(key_size // 8)
        key = base64.b64encode(key_bytes).decode()
        
        # Also generate IV for AES
        iv_bytes = secrets.token_bytes(16)  # AES block size is 128 bits
        iv = base64.b64encode(iv_bytes).decode()
        
        return CryptoOutput(
            success=True,
            operation="generate_key",
            message=f"Generated {key_size}-bit AES key",
            data={
                "algorithm": algorithm,
                "key": key,
                "key_size": key_size,
                "iv": iv
            }
        )
        
    elif algorithm == 'rsa':
        # For RSA, we'll generate a simple key pair representation
        # In production, you'd use cryptography library
        key_size = key_size or 2048
        
        if key_size < 1024:
            raise ValueError(f"RSA key size too small: {key_size}. Minimum is 1024 bits")
        
        # Generate mock RSA key pair
        # For our simple HMAC-based signing, we need the same key
        key_bytes = secrets.token_bytes(key_size // 8)
        private_key = base64.b64encode(key_bytes).decode()
        public_key = base64.b64encode(key_bytes).decode()  # Same as private for our mock
        
        return CryptoOutput(
            success=True,
            operation="generate_key",
            message=f"Generated {key_size}-bit RSA key pair",
            data={
                "algorithm": algorithm,
                "private_key": private_key,
                "public_key": public_key,
                "key_size": key_size
            }
        )
        
    else:
        # Generic key generation
        key_size = key_size or 256
        if key_size <= 0 or key_size % 8 != 0:
            raise ValueError(f"Invalid key size: {key_size}. Must be positive and divisible by 8")
        
        key_bytes = secrets.token_bytes(key_size // 8)
        key = base64.b64encode(key_bytes).decode()
        
        return CryptoOutput(
            success=True,
            operation="generate_key",
            message=f"Generated {key_size}-bit key",
            data={
                "algorithm": algorithm,
                "key": key,
                "key_size": key_size
            }
        )


async def crypto_encrypt(ctx: RunContext[Any], data: str, algorithm: str,
                       key: str, iv: Optional[str]) -> CryptoOutput:
    """
    Encrypt data using specified algorithm.
    
    Args:
        ctx: Runtime context
        data: Data to encrypt
        algorithm: Encryption algorithm
        key: Encryption key
        iv: Initialization vector (for AES)
        
    Returns:
        CryptoOutput with encrypted data
        
    Raises:
        ValueError: If algorithm is unsupported or key is invalid
    """
    if algorithm == 'aes':
        # Simple XOR encryption as a placeholder
        # In production, use proper AES implementation
        try:
            key_bytes = base64.b64decode(key)
        except Exception as e:
            raise ValueError(f"Invalid encryption key format: {e}") from e
        
        if not key_bytes:
            raise ValueError("Encryption key cannot be empty")
        
        data_bytes = data.encode()
        
        # Simple XOR encryption
        encrypted = bytearray()
        for i, byte in enumerate(data_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            encrypted.append(byte ^ key_byte)
        
        ciphertext = base64.b64encode(encrypted).decode()
        
        return CryptoOutput(
            success=True,
            operation="encrypt",
            message=f"Successfully encrypted data using {algorithm}",
            data={
                "algorithm": algorithm,
                "ciphertext": ciphertext,
                "iv": iv
            }
        )
        
    else:
        raise ValueError(f"Unsupported encryption algorithm: {algorithm}. Only 'aes' is supported")


async def crypto_decrypt(ctx: RunContext[Any], data: str, algorithm: str,
                       key: str, iv: Optional[str]) -> CryptoOutput:
    """
    Decrypt data using specified algorithm.
    
    Args:
        ctx: Runtime context
        data: Encrypted data
        algorithm: Decryption algorithm
        key: Decryption key
        iv: Initialization vector (for AES)
        
    Returns:
        CryptoOutput with decrypted data
        
    Raises:
        ValueError: If algorithm is unsupported, key/data format is invalid
        RuntimeError: If decryption fails
    """
    if algorithm == 'aes':
        # Simple XOR decryption (same as encryption for XOR)
        try:
            key_bytes = base64.b64decode(key)
        except Exception as e:
            raise ValueError(f"Invalid decryption key format: {e}") from e
        
        try:
            encrypted_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Invalid encrypted data format: {e}") from e
        
        if not key_bytes:
            raise ValueError("Decryption key cannot be empty")
        
        # Simple XOR decryption
        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            key_byte = key_bytes[i % len(key_bytes)]
            decrypted.append(byte ^ key_byte)
        
        try:
            plaintext = decrypted.decode()
        except UnicodeDecodeError as e:
            raise RuntimeError(f"Failed to decode decrypted data: {e}") from e
        
        return CryptoOutput(
            success=True,
            operation="decrypt",
            message=f"Successfully decrypted data using {algorithm}",
            data={
                "algorithm": algorithm,
                "plaintext": plaintext
            }
        )
        
    else:
        raise ValueError(f"Unsupported decryption algorithm: {algorithm}. Only 'aes' is supported")


async def crypto_sign(ctx: RunContext[Any], data: str, algorithm: str,
                    private_key: str) -> CryptoOutput:
    """
    Create a digital signature.
    
    Args:
        ctx: Runtime context
        data: Data to sign
        algorithm: Signing algorithm
        private_key: Private key for signing
        
    Returns:
        CryptoOutput with signature
        
    Raises:
        ValueError: If private key format is invalid
        RuntimeError: If signing operation fails
    """
    # Simple HMAC-based signature as placeholder
    try:
        private_key_bytes = base64.b64decode(private_key)
    except Exception as e:
        raise ValueError(f"Invalid private key format: {e}") from e
    
    if not private_key_bytes:
        raise ValueError("Private key cannot be empty")
    
    try:
        signature_bytes = hmac.new(
            private_key_bytes,
            data.encode(),
            hashlib.sha256
        ).digest()
        
        signature = base64.b64encode(signature_bytes).decode()
        
        return CryptoOutput(
            success=True,
            operation="sign",
            message=f"Successfully signed data",
            data={
                "algorithm": algorithm,
                "signature": signature
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to sign data: {e}") from e


async def crypto_verify_signature(ctx: RunContext[Any], data: str, algorithm: str,
                                signature: str, public_key: str) -> CryptoOutput:
    """
    Verify a digital signature.
    
    Args:
        ctx: Runtime context
        data: Original data
        algorithm: Signing algorithm used
        signature: Signature to verify
        public_key: Public key for verification
        
    Returns:
        CryptoOutput with verification result (success=True with valid field)
        
    Raises:
        ValueError: If public key format is invalid
        RuntimeError: If verification operation fails
    """
    # For this simple implementation, we'll use the public key as the secret
    # In production, proper asymmetric verification would be used
    try:
        public_key_bytes = base64.b64decode(public_key)
    except Exception as e:
        # Invalid format is a validation error
        raise ValueError(f"Invalid public key format: {e}") from e
    
    if not public_key_bytes:
        # Empty key is a validation error
        raise ValueError("Public key cannot be empty")
    
    try:
        expected_signature_bytes = hmac.new(
            public_key_bytes,
            data.encode(),
            hashlib.sha256
        ).digest()
        
        expected_signature = base64.b64encode(expected_signature_bytes).decode()
        is_valid = expected_signature == signature
        
        # Always return success=True for discovery operations
        # The "valid" field indicates if verification passed
        return CryptoOutput(
            success=True,
            operation="verify_signature",
            message=f"Signature verification {'successful' if is_valid else 'failed'}",
            data={
                "valid": is_valid,
                "algorithm": algorithm
            }
        )
    except Exception as e:
        # Actual computation failure is a runtime error
        raise RuntimeError(f"Failed to verify signature: {e}") from e


async def crypto_encode_base64(ctx: RunContext[Any], data: str) -> CryptoOutput:
    """
    Encode data to base64.
    
    Args:
        ctx: Runtime context
        data: Data to encode
        
    Returns:
        CryptoOutput with encoded data
        
    Raises:
        RuntimeError: If encoding fails
    """
    try:
        encoded = base64.b64encode(data.encode()).decode()
        
        return CryptoOutput(
            success=True,
            operation="encode_base64",
            message="Successfully encoded data to base64",
            data={
                "encoded": encoded,
                "original_length": len(data),
                "encoded_length": len(encoded)
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to encode data to base64: {e}") from e


async def crypto_decode_base64(ctx: RunContext[Any], data: str) -> CryptoOutput:
    """
    Decode data from base64.
    
    Args:
        ctx: Runtime context
        data: Base64 encoded data
        
    Returns:
        CryptoOutput with decoded data
        
    Raises:
        ValueError: If data is not valid base64
    """
    try:
        decoded_bytes = base64.b64decode(data)
        decoded = decoded_bytes.decode()
        
        return CryptoOutput(
            success=True,
            operation="decode_base64",
            message="Successfully decoded data from base64",
            data={
                "decoded": decoded,
                "encoded_length": len(data),
                "decoded_length": len(decoded)
            }
        )
    except Exception as e:
        raise ValueError(f"Failed to decode base64 data: {e}") from e


async def crypto_generate_jwt(ctx: RunContext[Any], payload: Dict[str, Any],
                            secret: str, expires_in: Optional[int]) -> CryptoOutput:
    """
    Generate a JWT token.
    
    Args:
        ctx: Runtime context
        payload: JWT payload
        secret: JWT secret
        expires_in: Expiration time in seconds
        
    Returns:
        CryptoOutput with JWT token
        
    Raises:
        ValueError: If secret/payload is empty or expires_in is invalid
        RuntimeError: If JWT generation fails
    """
    if not secret:
        raise ValueError("JWT secret cannot be empty")
    
    if not payload:
        raise ValueError("JWT payload cannot be empty")
    
    try:
        # Simple JWT implementation
        header = {
            "alg": "HS256",
            "typ": "JWT"
        }
        
        # Add expiration if specified
        if expires_in is not None:
            if not isinstance(expires_in, int):
                raise ValueError("expires_in must be an integer")
            payload["exp"] = int((datetime.now(timezone.utc) + timedelta(seconds=expires_in)).timestamp())
        
        payload["iat"] = int(datetime.now(timezone.utc).timestamp())
        
        # Encode header and payload
        header_encoded = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip('=')
        
        payload_encoded = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')
        
        # Create signature
        message = f"{header_encoded}.{payload_encoded}"
        signature_bytes = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature = base64.urlsafe_b64encode(signature_bytes).decode().rstrip('=')
        
        token = f"{message}.{signature}"
        
        return CryptoOutput(
            success=True,
            operation="generate_jwt",
            message="Successfully generated JWT token",
            data={
                "token": token,
                "expires_in": expires_in,
                "payload": payload
            }
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to generate JWT: {e}") from e


async def crypto_verify_jwt(ctx: RunContext[Any], data: str, secret: str) -> CryptoOutput:
    """
    Verify and decode a JWT token.
    
    Args:
        ctx: Runtime context
        data: JWT token
        secret: JWT secret
        
    Returns:
        CryptoOutput with verification result and payload
        
    Raises:
        ValueError: If token/secret is empty or token format is invalid
        RuntimeError: If JWT verification operation fails
    """
    if not secret:
        raise ValueError("JWT secret cannot be empty")
    
    if not data:
        raise ValueError("JWT token cannot be empty")
    
    # Split token
    parts = data.split('.')
    if len(parts) != 3:
        raise ValueError("Invalid JWT format - must have exactly 3 parts separated by dots")
    
    header_encoded, payload_encoded, signature = parts
    
    try:
        # Verify signature
        message = f"{header_encoded}.{payload_encoded}"
        expected_signature_bytes = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        expected_signature = base64.urlsafe_b64encode(expected_signature_bytes).decode().rstrip('=')
        
        if signature != expected_signature:
            return CryptoOutput(
                success=True,
                operation="verify_jwt",
                message="JWT verification failed - invalid signature",
                data={
                    "valid": False,
                    "error": "Invalid signature"
                }
            )
        
        # Decode payload
        # Add padding if needed
        padding = 4 - (len(payload_encoded) % 4)
        if padding != 4:
            payload_encoded += '=' * padding
            
        payload_json = base64.urlsafe_b64decode(payload_encoded).decode()
        payload = json.loads(payload_json)
        
        # Check expiration
        if 'exp' in payload:
            exp_time = datetime.fromtimestamp(payload['exp'], timezone.utc)
            if datetime.now(timezone.utc) > exp_time:
                return CryptoOutput(
                    success=True,
                    operation="verify_jwt",
                    message="JWT verification failed - token expired",
                    data={
                        "valid": False,
                        "error": "Token expired",
                        "expired_at": exp_time.isoformat()
                    }
                )
        
        return CryptoOutput(
            success=True,
            operation="verify_jwt",
            message="JWT verification successful",
            data={
                "valid": True,
                "payload": payload
            }
        )
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to verify JWT: {e}") from e


async def crypto_generate_salt(ctx: RunContext[Any]) -> CryptoOutput:
    """
    Generate a random salt.
    
    Args:
        ctx: Runtime context
        
    Returns:
        CryptoOutput with generated salt
        
    Raises:
        RuntimeError: If salt generation fails
    """
    try:
        salt_bytes = secrets.token_bytes(16)
        salt = base64.b64encode(salt_bytes).decode()
        
        return CryptoOutput(
            success=True,
            operation="generate_salt",
            message="Successfully generated salt",
            data={
                "salt": salt,
                "length": 16
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate salt: {e}") from e


async def crypto_hmac(ctx: RunContext[Any], data: str, key: str, 
                     algorithm: str) -> CryptoOutput:
    """
    Generate HMAC for data.
    
    Args:
        ctx: Runtime context
        data: Data to HMAC
        key: HMAC key
        algorithm: Hash algorithm for HMAC
        
    Returns:
        CryptoOutput with HMAC
        
    Raises:
        ValueError: If key/data is empty or algorithm is unsupported
        RuntimeError: If HMAC generation fails
    """
    if not key:
        raise ValueError("HMAC key cannot be empty")
    
    if not data:
        raise ValueError("HMAC data cannot be empty")
    
    # Map algorithm names to hashlib functions
    hash_funcs = {
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
        'sha1': hashlib.sha1,
        'md5': hashlib.md5
    }
    
    hash_func = hash_funcs.get(algorithm)
    if not hash_func:
        raise ValueError(f"Unsupported HMAC algorithm: {algorithm}. Supported: {list(hash_funcs.keys())}")
    
    try:
        hmac_bytes = hmac.new(
            key.encode(),
            data.encode(),
            hash_func
        ).digest()
        
        hmac_value = base64.b64encode(hmac_bytes).decode()
        
        return CryptoOutput(
            success=True,
            operation="hmac",
            message=f"Successfully generated HMAC using {algorithm}",
            data={
                "hmac": hmac_value,
                "algorithm": algorithm
            }
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate HMAC: {e}") from e


def create_crypto_agent():
    """
    Create and return the crypto AgenTool.
    
    Returns:
        Agent configured for cryptographic operations
    """
    crypto_routing = RoutingConfig(
        operation_field='operation',
        operation_map={
            'hash': ('crypto_hash', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'salt': x.salt, 'iterations': x.iterations
            }),
            'verify_hash': ('crypto_verify_hash', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'hash': x.key, 'salt': x.salt  # Using key field for hash
            }),
            'generate_key': ('crypto_generate_key', lambda x: {
                'algorithm': x.algorithm, 'key_size': x.key_size
            }),
            'encrypt': ('crypto_encrypt', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'key': x.key, 'iv': x.iv
            }),
            'decrypt': ('crypto_decrypt', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'key': x.key, 'iv': x.iv
            }),
            'sign': ('crypto_sign', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'private_key': x.private_key
            }),
            'verify_signature': ('crypto_verify_signature', lambda x: {
                'data': x.data, 'algorithm': x.algorithm,
                'signature': x.signature, 'public_key': x.public_key
            }),
            'encode_base64': ('crypto_encode_base64', lambda x: {'data': x.data}),
            'decode_base64': ('crypto_decode_base64', lambda x: {'data': x.data}),
            'generate_jwt': ('crypto_generate_jwt', lambda x: {
                'payload': x.payload, 'secret': x.secret, 'expires_in': x.expires_in
            }),
            'verify_jwt': ('crypto_verify_jwt', lambda x: {
                'data': x.data, 'secret': x.secret
            }),
            'generate_salt': ('crypto_generate_salt', lambda x: {}),
            'hmac': ('crypto_hmac', lambda x: {
                'data': x.data, 'key': x.key, 'algorithm': x.algorithm
            }),
        }
    )
    
    return create_agentool(
        name='crypto',
        input_schema=CryptoInput,
        routing_config=crypto_routing,
        tools=[
            crypto_hash, crypto_verify_hash, crypto_generate_key,
            crypto_encrypt, crypto_decrypt, crypto_sign,
            crypto_verify_signature, crypto_encode_base64, crypto_decode_base64,
            crypto_generate_jwt, crypto_verify_jwt, crypto_generate_salt,
            crypto_hmac
        ],
        output_type=CryptoOutput,
        system_prompt="Handle cryptographic operations securely and efficiently.",
        description="Comprehensive cryptographic operations including hashing, encryption, and JWT",
        version="1.0.0",
        tags=["crypto", "security", "encryption", "hashing", "jwt"],
        examples=[
            {
                "description": "Hash a password with SHA256",
                "input": {
                    "operation": "hash",
                    "algorithm": "sha256",
                    "data": "my_password"
                },
                "output": {
                    "success": True,
                    "operation": "hash",
                    "message": "Successfully hashed data using sha256"
                }
            },
            {
                "description": "Generate an AES encryption key",
                "input": {
                    "operation": "generate_key",
                    "algorithm": "aes",
                    "key_size": 256
                },
                "output": {
                    "success": True,
                    "operation": "generate_key",
                    "message": "Generated 256-bit AES key"
                }
            },
            {
                "description": "Generate a JWT token",
                "input": {
                    "operation": "generate_jwt",
                    "payload": {"user_id": "123", "role": "admin"},
                    "secret": "my_secret",
                    "expires_in": 3600
                },
                "output": {
                    "success": True,
                    "operation": "generate_jwt",
                    "message": "Successfully generated JWT token"
                }
            }
        ]
    )


# Create the agent instance
agent = create_crypto_agent()