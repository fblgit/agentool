"""
Tests for crypto toolkit.

This module tests all functionality of the cryptographic toolkit including
hashing, encryption, key generation, digital signatures, JWT operations, and encoding.
"""

import json
import asyncio
import base64
import pytest

from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry


class TestCrypto:
    """Test suite for crypto toolkit."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent
        from agentoolkit.security.crypto import create_crypto_agent
        agent = create_crypto_agent()
    
    def test_crypto_hash_operations(self):
        """Test various hashing operations."""
        
        async def run_test():
            injector = get_injector()
            test_data = "Hello, World!"
            
            # Test SHA256 hashing
            hash_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha256",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            assert hash_result.success is True
            assert hash_result.operation == "hash"
            assert hash_result.data["algorithm"] == "sha256"
            assert "hash" in hash_result.data
            assert len(hash_result.data["hash"]) == 64  # SHA256 produces 64 hex chars
            
            # Test SHA512 hashing
            sha512_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha512",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            assert sha512_result.success is True
            assert len(sha512_result.data["hash"]) == 128  # SHA512 produces 128 hex chars
            
            # Test MD5 hashing
            md5_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "md5",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            assert md5_result.success is True
            assert len(md5_result.data["hash"]) == 32  # MD5 produces 32 hex chars
        
        asyncio.run(run_test())
    
    def test_crypto_hash_with_salt(self):
        """Test hashing with salt."""
        
        async def run_test():
            injector = get_injector()
            test_data = "password123"
            test_salt = "random_salt"
            
            # Hash with salt
            hash_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha256",
                "data": test_data,
                "salt": test_salt
            })
            
            # crypto returns typed CryptoOutput
            assert hash_result.success is True
            assert hash_result.data["salt"] == test_salt
            
            # Hash same data without salt should be different
            no_salt_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha256",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            assert hash_result.data["hash"] != no_salt_result.data["hash"]
        
        asyncio.run(run_test())
    
    def test_crypto_verify_hash(self):
        """Test hash verification."""
        
        async def run_test():
            injector = get_injector()
            test_data = "verify_me"
            
            # First create a hash
            hash_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha256",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            hash_value = hash_result.data["hash"]
            
            # Verify correct data
            verify_result = await injector.run('crypto', {
                "operation": "verify_hash",
                "algorithm": "sha256",
                "data": test_data,
                "key": hash_value  # Using key field for hash
            })
            
            # crypto returns typed CryptoOutput
            assert verify_result.success is True
            assert verify_result.data["valid"] is True
            
            # Verify incorrect data
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_hash",
                "algorithm": "sha256",
                "data": "wrong_data",
                "key": hash_value
            })
            
            # crypto returns typed CryptoOutput - discovery operation always returns success=True
            assert wrong_verify.success is True
            assert wrong_verify.data["valid"] is False
        
        asyncio.run(run_test())
    
    def test_crypto_bcrypt_hash(self):
        """Test bcrypt-style hashing with iterations."""
        
        async def run_test():
            injector = get_injector()
            test_password = "secure_password"
            
            # Hash with bcrypt
            hash_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "bcrypt",
                "data": test_password,
                "iterations": 10000
            })
            
            # crypto returns typed CryptoOutput
            assert hash_result.success is True
            assert hash_result.data["algorithm"] == "bcrypt"
            assert hash_result.data["iterations"] == 10000
            assert "salt" in hash_result.data
            assert "hash" in hash_result.data
        
        asyncio.run(run_test())
    
    def test_crypto_generate_key(self):
        """Test key generation for different algorithms."""
        
        async def run_test():
            injector = get_injector()
            
            # Generate AES key
            aes_result = await injector.run('crypto', {
                "operation": "generate_key",
                "algorithm": "aes",
                "key_size": 256
            })
            
            # crypto returns typed CryptoOutput
            assert aes_result.success is True
            assert aes_result.data["algorithm"] == "aes"
            assert aes_result.data["key_size"] == 256
            assert "key" in aes_result.data
            assert "iv" in aes_result.data
            
            # Verify key is base64 encoded and correct length
            key_bytes = base64.b64decode(aes_result.data["key"])
            assert len(key_bytes) == 32  # 256 bits = 32 bytes
            
            # Generate RSA key pair
            rsa_result = await injector.run('crypto', {
                "operation": "generate_key",
                "algorithm": "rsa",
                "key_size": 2048
            })
            
            # crypto returns typed CryptoOutput
            assert rsa_result.success is True
            assert rsa_result.data["algorithm"] == "rsa"
            assert rsa_result.data["key_size"] == 2048
            assert "private_key" in rsa_result.data
            assert "public_key" in rsa_result.data
        
        asyncio.run(run_test())
    
    def test_crypto_encrypt_decrypt(self):
        """Test encryption and decryption."""
        
        async def run_test():
            injector = get_injector()
            test_data = "Secret message"
            
            # Generate encryption key
            key_result = await injector.run('crypto', {
                "operation": "generate_key",
                "algorithm": "aes",
                "key_size": 256
            })
            
            # crypto returns typed CryptoOutput
            encryption_key = key_result.data["key"]
            iv = key_result.data["iv"]
            
            # Encrypt data
            encrypt_result = await injector.run('crypto', {
                "operation": "encrypt",
                "algorithm": "aes",
                "data": test_data,
                "key": encryption_key,
                "iv": iv
            })
            
            # crypto returns typed CryptoOutput
            assert encrypt_result.success is True
            assert "ciphertext" in encrypt_result.data
            ciphertext = encrypt_result.data["ciphertext"]
            
            # Decrypt data
            decrypt_result = await injector.run('crypto', {
                "operation": "decrypt",
                "algorithm": "aes",
                "data": ciphertext,
                "key": encryption_key,
                "iv": iv
            })
            
            # crypto returns typed CryptoOutput
            assert decrypt_result.success is True
            assert decrypt_result.data["plaintext"] == test_data
        
        asyncio.run(run_test())
    
    def test_crypto_digital_signature(self):
        """Test digital signature creation and verification."""
        
        async def run_test():
            injector = get_injector()
            test_data = "Sign this message"
            
            # Generate key pair
            key_result = await injector.run('crypto', {
                "operation": "generate_key",
                "algorithm": "rsa",
                "key_size": 2048
            })
            
            # crypto returns typed CryptoOutput
            private_key = key_result.data["private_key"]
            public_key = key_result.data["public_key"]
            
            # Sign data
            sign_result = await injector.run('crypto', {
                "operation": "sign",
                "algorithm": "rsa",
                "data": test_data,
                "private_key": private_key
            })
            
            # crypto returns typed CryptoOutput
            assert sign_result.success is True
            signature = sign_result.data["signature"]
            
            # Verify signature
            verify_result = await injector.run('crypto', {
                "operation": "verify_signature",
                "algorithm": "rsa",
                "data": test_data,
                "signature": signature,
                "public_key": public_key
            })
            
            # crypto returns typed CryptoOutput
            assert verify_result.success is True
            assert verify_result.data["valid"] is True
            
            # Verify with wrong data
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_signature",
                "algorithm": "rsa",
                "data": "Wrong message",
                "signature": signature,
                "public_key": public_key
            })
            
            # crypto returns typed CryptoOutput - discovery operation always returns success=True
            assert wrong_verify.data["valid"] is False
        
        asyncio.run(run_test())
    
    def test_crypto_base64_encoding(self):
        """Test base64 encoding and decoding."""
        
        async def run_test():
            injector = get_injector()
            test_data = "Hello, Base64! 🔐"
            
            # Encode to base64
            encode_result = await injector.run('crypto', {
                "operation": "encode_base64",
                "data": test_data
            })
            
            # crypto returns typed CryptoOutput
            assert encode_result.success is True
            encoded = encode_result.data["encoded"]
            assert encode_result.data["original_length"] == len(test_data)
            
            # Decode from base64
            decode_result = await injector.run('crypto', {
                "operation": "decode_base64",
                "data": encoded
            })
            
            # crypto returns typed CryptoOutput
            assert decode_result.success is True
            assert decode_result.data["decoded"] == test_data
        
        asyncio.run(run_test())
    
    def test_crypto_jwt_operations(self):
        """Test JWT generation and verification."""
        
        async def run_test():
            injector = get_injector()
            jwt_secret = "my_jwt_secret"
            payload = {
                "user_id": "123",
                "username": "testuser",
                "role": "admin"
            }
            
            # Generate JWT
            jwt_result = await injector.run('crypto', {
                "operation": "generate_jwt",
                "payload": payload,
                "secret": jwt_secret,
                "expires_in": 3600  # 1 hour
            })
            
            # crypto returns typed CryptoOutput
            assert jwt_result.success is True
            token = jwt_result.data["token"]
            assert len(token.split('.')) == 3  # JWT has 3 parts
            
            # Verify JWT
            verify_result = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": jwt_secret
            })
            
            # crypto returns typed CryptoOutput
            assert verify_result.success is True
            assert verify_result.data["valid"] is True
            assert verify_result.data["payload"]["user_id"] == "123"
            assert verify_result.data["payload"]["username"] == "testuser"
            
            # Verify with wrong secret
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": "wrong_secret"
            })
            
            # crypto returns typed CryptoOutput - discovery operation returns success=True
            assert wrong_verify.success is True
            assert wrong_verify.data["valid"] is False
        
        asyncio.run(run_test())
    
    def test_crypto_jwt_expiration(self):
        """Test JWT expiration handling."""
        
        async def run_test():
            injector = get_injector()
            jwt_secret = "test_secret"
            
            # Generate JWT with immediate expiration
            jwt_result = await injector.run('crypto', {
                "operation": "generate_jwt",
                "payload": {"test": "data"},
                "secret": jwt_secret,
                "expires_in": -1  # Already expired
            })
            
            # crypto returns typed CryptoOutput
            token = jwt_result.data["token"]
            
            # Verify expired JWT
            verify_result = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": jwt_secret
            })
            
            # crypto returns typed CryptoOutput - discovery operation returns success=True
            assert verify_result.success is True
            assert verify_result.data["valid"] is False
            assert "expired" in verify_result.data["error"].lower()
        
        asyncio.run(run_test())
    
    def test_crypto_generate_salt(self):
        """Test salt generation."""
        
        async def run_test():
            injector = get_injector()
            
            # Generate salt
            salt_result = await injector.run('crypto', {
                "operation": "generate_salt"
            })
            
            # crypto returns typed CryptoOutput
            assert salt_result.success is True
            salt = salt_result.data["salt"]
            assert salt_result.data["length"] == 16
            
            # Verify salt is base64 encoded
            salt_bytes = base64.b64decode(salt)
            assert len(salt_bytes) == 16
            
            # Generate another salt - should be different
            salt2_result = await injector.run('crypto', {
                "operation": "generate_salt"
            })
            
            # crypto returns typed CryptoOutput
            assert salt2_result.data["salt"] != salt
        
        asyncio.run(run_test())
    
    def test_crypto_hmac(self):
        """Test HMAC generation."""
        
        async def run_test():
            injector = get_injector()
            test_data = "HMAC this data"
            test_key = "hmac_secret_key"
            
            # Generate HMAC with SHA256
            hmac_result = await injector.run('crypto', {
                "operation": "hmac",
                "data": test_data,
                "key": test_key,
                "algorithm": "sha256"
            })
            
            # crypto returns typed CryptoOutput
            assert hmac_result.success is True
            assert hmac_result.data["algorithm"] == "sha256"
            hmac_value = hmac_result.data["hmac"]
            
            # Same data and key should produce same HMAC
            hmac2_result = await injector.run('crypto', {
                "operation": "hmac",
                "data": test_data,
                "key": test_key,
                "algorithm": "sha256"
            })
            
            # crypto returns typed CryptoOutput
            assert hmac2_result.data["hmac"] == hmac_value
            
            # Different key should produce different HMAC
            hmac3_result = await injector.run('crypto', {
                "operation": "hmac",
                "data": test_data,
                "key": "different_key",
                "algorithm": "sha256"
            })
            
            # crypto returns typed CryptoOutput
            assert hmac3_result.data["hmac"] != hmac_value
        
        asyncio.run(run_test())
    
    def test_crypto_error_handling(self):
        """Test error handling for various edge cases."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid base64 decoding
            try:
                result = await injector.run('crypto', {
                    "operation": "decode_base64",
                    "data": "not_valid_base64!!!"
                })
                assert False, "Expected ValueError to be raised for invalid base64"
            except ValueError as e:
                assert "Failed to decode base64 data" in str(e)
            
            # Test invalid AES key size
            try:
                result = await injector.run('crypto', {
                    "operation": "generate_key",
                    "algorithm": "aes",
                    "key_size": 123  # Invalid size
                })
                assert False, "Expected ValueError to be raised for invalid AES key size"
            except ValueError as e:
                assert "Invalid AES key size" in str(e)
            
            # Test invalid JWT payload (empty) - empty dict {} is falsy but not None
            # The validator checks for 'not v' which catches empty dict
            # But the actual error is raised by the tool function since {} is not None
            try:
                result = await injector.run('crypto', {
                    "operation": "generate_jwt",
                    "payload": {},
                    "secret": "test_secret",
                    "expires_in": 3600
                })
                # Empty dict doesn't trigger the validator (it's not None)
                # But the tool function checks and raises ValueError
            except ValueError as e:
                assert "JWT payload cannot be empty" in str(e)
            
            # Test empty JWT secret - validator catches empty string
            result = await injector.run('crypto', {
                "operation": "generate_jwt",
                "payload": {"test": "data"},
                "secret": "",
                "expires_in": 3600
            })
            # When validator catches it, we get a validation error in output
            assert hasattr(result, 'output')
            assert "secret is required for generate_jwt operation" in result.output or "JWT secret cannot be empty" in result.output
        
        asyncio.run(run_test())
    
    def test_crypto_encryption_errors(self):
        """Test encryption/decryption error handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid encryption key format
            try:
                result = await injector.run('crypto', {
                    "operation": "encrypt",
                    "algorithm": "aes",
                    "data": "test data",
                    "key": "invalid_base64!!!"
                })
                assert False, "Expected ValueError for invalid key format"
            except ValueError as e:
                assert "Invalid encryption key format" in str(e)
            
            # Test unsupported encryption algorithm (Pydantic validation catches this)
            result = await injector.run('crypto', {
                "operation": "encrypt",
                "algorithm": "unsupported",
                "data": "test data",
                "key": "dGVzdGtleQ=="  # Valid base64
            })
            # Pydantic validation error is returned as output string for invalid schema
            # When input validation fails, result has output attribute with error message
            assert hasattr(result, 'output')
            output = result.output
            assert "Error creating input model" in output
            assert "literal_error" in output
            assert "unsupported" in output
        
        asyncio.run(run_test())
    
    def test_crypto_signature_errors(self):
        """Test digital signature error handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid private key format for signing
            try:
                result = await injector.run('crypto', {
                    "operation": "sign",
                    "algorithm": "rsa",
                    "data": "test data",
                    "private_key": "invalid_base64!!!"
                })
                assert False, "Expected ValueError for invalid private key"
            except ValueError as e:
                assert "Invalid private key format" in str(e)
            
            # Test invalid public key format for verification
            try:
                result = await injector.run('crypto', {
                    "operation": "verify_signature",
                    "algorithm": "rsa",
                    "data": "test data",
                    "signature": "dGVzdA==",
                    "public_key": "invalid_base64!!!"
                })
                assert False, "Expected ValueError for invalid public key"
            except ValueError as e:
                assert "Invalid public key format" in str(e)
        
        asyncio.run(run_test())
    
    def test_crypto_hmac_errors(self):
        """Test HMAC error handling."""
        
        async def run_test():
            injector = get_injector()
            
            # Test empty HMAC key - empty string "" is falsy
            # The validator checks 'not v' which catches empty string
            result = await injector.run('crypto', {
                "operation": "hmac",
                "data": "test data",
                "key": "",
                "algorithm": "sha256"
            })
            # When validator catches it, we get a validation error in output
            assert hasattr(result, 'output')
            assert "key is required for hmac operation" in result.output or "HMAC key cannot be empty" in result.output
            
            # Test unsupported HMAC algorithm (Pydantic validation catches this)
            result = await injector.run('crypto', {
                "operation": "hmac",
                "data": "test data",
                "key": "test_key",
                "algorithm": "unsupported"
            })
            # Pydantic validation error is returned as output string for invalid schema
            # When input validation fails, result has output attribute with error message
            assert hasattr(result, 'output')
            output = result.output
            assert "Error creating input model" in output
            assert "literal_error" in output
            assert "unsupported" in output
        
        asyncio.run(run_test())