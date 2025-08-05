# Crypto AgenToolkit

> **Note**: For guidelines on creating new AgenToolkits, see [CRAFTING_AGENTOOLS.md](../CRAFTING_AGENTOOLS.md). For testing patterns and examples, refer to [tests/agentoolkit/test_crypto.py](../../../tests/agentoolkit/test_crypto.py).

## Overview

The Crypto AgenToolkit provides comprehensive cryptographic operations including hashing, encryption, digital signatures, JWT management, and encoding utilities. It offers a secure foundation for authentication, data protection, and cryptographic key management.

### Key Features
- Multiple hashing algorithms (SHA256, SHA512, MD5, SHA1, bcrypt)
- Symmetric encryption (AES with XOR placeholder)
- Asymmetric operations (RSA key generation, digital signatures)
- JWT token generation and verification
- HMAC generation for message authentication
- Base64 encoding/decoding
- Cryptographic key and salt generation
- Hash verification with salt support

## Creation Method

```python
from agentoolkit.security.crypto import create_crypto_agent

# Create the agent
agent = create_crypto_agent()
```

The creation function returns a fully configured AgenTool with name `'crypto'`.

## Input Schema

### CryptoInput

The input schema is a Pydantic model with the following fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `operation` | `Literal['hash', 'verify_hash', 'generate_key', 'encrypt', 'decrypt', 'sign', 'verify_signature', 'encode_base64', 'decode_base64', 'generate_jwt', 'verify_jwt', 'generate_salt', 'hmac']` | Yes | - | The cryptographic operation to perform |
| `data` | `Optional[str]` | No | None | Data to process |
| `algorithm` | `Optional[Literal['sha256', 'sha512', 'md5', 'sha1', 'bcrypt', 'aes', 'rsa']]` | No | None | Algorithm to use |
| `key` | `Optional[str]` | No | None | Encryption/decryption key or hash for verification |
| `salt` | `Optional[str]` | No | None | Salt for hashing |
| `iterations` | `Optional[int]` | No | None | Iterations for key derivation (bcrypt) |
| `iv` | `Optional[str]` | No | None | Initialization vector for AES |
| `key_size` | `Optional[int]` | No | None | Key size in bits |
| `payload` | `Optional[Dict[str, Any]]` | No | None | JWT payload |
| `secret` | `Optional[str]` | No | None | JWT secret |
| `expires_in` | `Optional[int]` | No | None | JWT expiration time in seconds |
| `private_key` | `Optional[str]` | No | None | Private key for signing |
| `public_key` | `Optional[str]` | No | None | Public key for verification |
| `signature` | `Optional[str]` | No | None | Signature to verify |

## Operations Schema

The routing configuration maps operations to tool functions:

| Operation | Tool Function | Required Parameters | Description |
|-----------|--------------|-------------------|-------------|
| `hash` | `crypto_hash` | `data`, `algorithm`, `salt`, `iterations` | Hash data using specified algorithm |
| `verify_hash` | `crypto_verify_hash` | `data`, `algorithm`, `key` (as hash), `salt` | Verify a hash against data |
| `generate_key` | `crypto_generate_key` | `algorithm`, `key_size` | Generate cryptographic keys |
| `encrypt` | `crypto_encrypt` | `data`, `algorithm`, `key`, `iv` | Encrypt data |
| `decrypt` | `crypto_decrypt` | `data`, `algorithm`, `key`, `iv` | Decrypt data |
| `sign` | `crypto_sign` | `data`, `algorithm`, `private_key` | Create digital signature |
| `verify_signature` | `crypto_verify_signature` | `data`, `algorithm`, `signature`, `public_key` | Verify digital signature |
| `encode_base64` | `crypto_encode_base64` | `data` | Encode to base64 |
| `decode_base64` | `crypto_decode_base64` | `data` | Decode from base64 |
| `generate_jwt` | `crypto_generate_jwt` | `payload`, `secret`, `expires_in` | Generate JWT token |
| `verify_jwt` | `crypto_verify_jwt` | `data`, `secret` | Verify and decode JWT |
| `generate_salt` | `crypto_generate_salt` | - | Generate random salt |
| `hmac` | `crypto_hmac` | `data`, `key`, `algorithm` | Generate HMAC |

## Output Schema

### CryptoOutput

All operations return a `CryptoOutput` model with:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the operation succeeded |
| `operation` | `str` | The operation that was performed |
| `message` | `str` | Human-readable result message |
| `data` | `Optional[Any]` | Operation-specific data |
| `error` | `Optional[str]` | Error message if operation failed |

### Operation-Specific Data Fields

- **hash**: `algorithm`, `hash`, `salt`, `iterations` (for bcrypt)
- **verify_hash**: `valid`, `algorithm`
- **generate_key**: `algorithm`, `key`, `key_size`, `iv` (for AES), `private_key`/`public_key` (for RSA)
- **encrypt**: `algorithm`, `ciphertext`, `iv`
- **decrypt**: `algorithm`, `plaintext`
- **sign**: `algorithm`, `signature`
- **verify_signature**: `valid`, `algorithm`
- **encode_base64**: `encoded`, `original_length`, `encoded_length`
- **decode_base64**: `decoded`, `encoded_length`, `decoded_length`
- **generate_jwt**: `token`, `expires_in`, `payload`
- **verify_jwt**: `valid`, `payload` or `error`, `expired_at` (if expired)
- **generate_salt**: `salt`, `length`
- **hmac**: `hmac`, `algorithm`

## Dependencies

This AgenToolkit has no external dependencies on other AgenToolkits. It uses Python's built-in hashlib, hmac, base64, secrets, and json modules for cryptographic operations.

## Tools

### crypto_hash
```python
async def crypto_hash(ctx: RunContext[Any], data: str, algorithm: str, salt: Optional[str], iterations: Optional[int]) -> CryptoOutput
```
Hash data using the specified algorithm. For bcrypt, uses PBKDF2 with SHA256.

**Raises:**
- `ValueError`: If algorithm is not supported
- `RuntimeError`: If hashing operation fails

### crypto_verify_hash
```python
async def crypto_verify_hash(ctx: RunContext[Any], data: str, algorithm: str, hash: str, salt: Optional[str]) -> CryptoOutput
```
Verify a hash against data. For bcrypt, requires salt parameter.

**Raises:**
- `ValueError`: If required parameters are missing or invalid
- `RuntimeError`: If hash verification operation fails

### crypto_generate_key
```python
async def crypto_generate_key(ctx: RunContext[Any], algorithm: str, key_size: Optional[int]) -> CryptoOutput
```
Generate cryptographic keys. For AES, also generates an IV. For RSA, generates a key pair.

**Raises:**
- `ValueError`: If algorithm or key size is invalid

### crypto_encrypt
```python
async def crypto_encrypt(ctx: RunContext[Any], data: str, algorithm: str, key: str, iv: Optional[str]) -> CryptoOutput
```
Encrypt data using XOR cipher (placeholder for AES).

**Raises:**
- `ValueError`: If algorithm is unsupported or key is invalid

### crypto_decrypt
```python
async def crypto_decrypt(ctx: RunContext[Any], data: str, algorithm: str, key: str, iv: Optional[str]) -> CryptoOutput
```
Decrypt data using XOR cipher (placeholder for AES).

**Raises:**
- `ValueError`: If algorithm is unsupported, key/data format is invalid
- `RuntimeError`: If decryption fails

### crypto_sign
```python
async def crypto_sign(ctx: RunContext[Any], data: str, algorithm: str, private_key: str) -> CryptoOutput
```
Create a digital signature using HMAC-SHA256.

**Raises:**
- `ValueError`: If private key format is invalid
- `RuntimeError`: If signing operation fails

### crypto_verify_signature
```python
async def crypto_verify_signature(ctx: RunContext[Any], data: str, algorithm: str, signature: str, public_key: str) -> CryptoOutput
```
Verify a digital signature using HMAC-SHA256.

**Raises:**
- `ValueError`: If public key format is invalid
- `RuntimeError`: If verification operation fails

### crypto_generate_jwt
```python
async def crypto_generate_jwt(ctx: RunContext[Any], payload: Dict[str, Any], secret: str, expires_in: Optional[int]) -> CryptoOutput
```
Generate a JWT token with HS256 algorithm.

**Raises:**
- `ValueError`: If secret/payload is empty or expires_in is invalid
- `RuntimeError`: If JWT generation fails

### crypto_verify_jwt
```python
async def crypto_verify_jwt(ctx: RunContext[Any], data: str, secret: str) -> CryptoOutput
```
Verify and decode a JWT token, checking signature and expiration.

**Raises:**
- `ValueError`: If token/secret is empty or token format is invalid
- `RuntimeError`: If JWT verification operation fails

### crypto_generate_salt
```python
async def crypto_generate_salt(ctx: RunContext[Any]) -> CryptoOutput
```
Generate a 16-byte random salt, base64 encoded.

**Raises:**
- `RuntimeError`: If salt generation fails

### crypto_hmac
```python
async def crypto_hmac(ctx: RunContext[Any], data: str, key: str, algorithm: str) -> CryptoOutput
```
Generate HMAC for data using specified hash algorithm.

**Raises:**
- `ValueError`: If key/data is empty or algorithm is unsupported
- `RuntimeError`: If HMAC generation fails

## Exceptions

| Exception Type | Scenarios |
|---------------|-----------|
| `ValueError` | - Unsupported algorithm<br>- Invalid key size<br>- Missing required parameters<br>- Invalid base64 data<br>- Invalid JWT format |
| `RuntimeError` | - Cryptographic operation failures<br>- Encoding/decoding errors<br>- Key generation failures |
| `UnicodeDecodeError` | - Invalid text encoding after decryption |

## Usage Examples

### Hashing and Verification
```python
from agentoolkit.security.crypto import create_crypto_agent
from agentool.core.injector import get_injector

# Create and register the agent
agent = create_crypto_agent()
injector = get_injector()

# Hash a password
result = await injector.run('crypto', {
    "operation": "hash",
    "algorithm": "sha256",
    "data": "my_password",
    "salt": "random_salt"
})

# Verify the hash
result = await injector.run('crypto', {
    "operation": "verify_hash",
    "algorithm": "sha256",
    "data": "my_password",
    "key": stored_hash,  # The hash to verify against
    "salt": "random_salt"
})
```

### Encryption and Decryption
```python
# Generate AES key
result = await injector.run('crypto', {
    "operation": "generate_key",
    "algorithm": "aes",
    "key_size": 256
})
key = result.data["key"]
iv = result.data["iv"]

# Encrypt data
result = await injector.run('crypto', {
    "operation": "encrypt",
    "algorithm": "aes",
    "data": "secret message",
    "key": key,
    "iv": iv
})

# Decrypt data
result = await injector.run('crypto', {
    "operation": "decrypt",
    "algorithm": "aes",
    "data": ciphertext,
    "key": key,
    "iv": iv
})
```

### JWT Operations
```python
# Generate JWT token
result = await injector.run('crypto', {
    "operation": "generate_jwt",
    "payload": {"user_id": "123", "role": "admin"},
    "secret": "my_secret_key",
    "expires_in": 3600  # 1 hour
})

# Verify JWT token
result = await injector.run('crypto', {
    "operation": "verify_jwt",
    "data": token,
    "secret": "my_secret_key"
})
```

### Digital Signatures
```python
# Generate RSA key pair
result = await injector.run('crypto', {
    "operation": "generate_key",
    "algorithm": "rsa",
    "key_size": 2048
})
private_key = result.data["private_key"]
public_key = result.data["public_key"]

# Sign data
result = await injector.run('crypto', {
    "operation": "sign",
    "algorithm": "rsa",
    "data": "important document",
    "private_key": private_key
})

# Verify signature
result = await injector.run('crypto', {
    "operation": "verify_signature",
    "algorithm": "rsa",
    "data": "important document",
    "signature": signature,
    "public_key": public_key
})
```

## Testing

The test suite is located at `tests/agentoolkit/test_crypto.py`. Tests cover:
- All hashing algorithms
- Hash verification with salts
- Key generation for different algorithms
- Encryption and decryption
- Digital signatures
- JWT generation and verification
- Base64 encoding/decoding
- HMAC generation
- Error conditions and edge cases

To run tests:
```bash
pytest tests/agentoolkit/test_crypto.py -v
```

## Notes

- This implementation uses simplified cryptography for demonstration (XOR for AES, HMAC for RSA signatures)
- In production, use proper cryptographic libraries (cryptography, pycryptodome)
- bcrypt implementation uses PBKDF2 with SHA256 as a placeholder
- JWT implementation follows HS256 algorithm specification
- All keys and encrypted data are base64 encoded for safe string handling
- Default iterations for bcrypt is 100000 (configurable)
- Salt generation produces 16-byte random values
- JWT tokens include `iat` (issued at) claim automatically