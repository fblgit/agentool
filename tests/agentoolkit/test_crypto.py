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
            
            if hasattr(hash_result, 'output'):
                hash_data = json.loads(hash_result.output)
            else:
                hash_data = hash_result
            
            print(f"Hash result: {hash_data}")
            assert hash_data["success"] is True
            assert hash_data["operation"] == "hash"
            assert hash_data["data"]["algorithm"] == "sha256"
            assert "hash" in hash_data["data"]
            assert len(hash_data["data"]["hash"]) == 64  # SHA256 produces 64 hex chars
            
            # Test SHA512 hashing
            sha512_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha512",
                "data": test_data
            })
            
            if hasattr(sha512_result, 'output'):
                sha512_data = json.loads(sha512_result.output)
            else:
                sha512_data = sha512_result
            
            assert sha512_data["success"] is True
            assert len(sha512_data["data"]["hash"]) == 128  # SHA512 produces 128 hex chars
            
            # Test MD5 hashing
            md5_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "md5",
                "data": test_data
            })
            
            if hasattr(md5_result, 'output'):
                md5_data = json.loads(md5_result.output)
            else:
                md5_data = md5_result
            
            assert md5_data["success"] is True
            assert len(md5_data["data"]["hash"]) == 32  # MD5 produces 32 hex chars
        
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
            
            if hasattr(hash_result, 'output'):
                hash_data = json.loads(hash_result.output)
            else:
                hash_data = hash_result
            
            assert hash_data["success"] is True
            assert hash_data["data"]["salt"] == test_salt
            
            # Hash same data without salt should be different
            no_salt_result = await injector.run('crypto', {
                "operation": "hash",
                "algorithm": "sha256",
                "data": test_data
            })
            
            if hasattr(no_salt_result, 'output'):
                no_salt_data = json.loads(no_salt_result.output)
            else:
                no_salt_data = no_salt_result
            
            assert hash_data["data"]["hash"] != no_salt_data["data"]["hash"]
        
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
            
            if hasattr(hash_result, 'output'):
                hash_data = json.loads(hash_result.output)
            else:
                hash_data = hash_result
            
            hash_value = hash_data["data"]["hash"]
            
            # Verify correct data
            verify_result = await injector.run('crypto', {
                "operation": "verify_hash",
                "algorithm": "sha256",
                "data": test_data,
                "key": hash_value  # Using key field for hash
            })
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            assert verify_data["success"] is True
            assert verify_data["data"]["valid"] is True
            
            # Verify incorrect data
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_hash",
                "algorithm": "sha256",
                "data": "wrong_data",
                "key": hash_value
            })
            
            if hasattr(wrong_verify, 'output'):
                wrong_data = json.loads(wrong_verify.output)
            else:
                wrong_data = wrong_verify
            
            assert wrong_data["success"] is True
            assert wrong_data["data"]["valid"] is False
        
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
            
            if hasattr(hash_result, 'output'):
                hash_data = json.loads(hash_result.output)
            else:
                hash_data = hash_result
            
            assert hash_data["success"] is True
            assert hash_data["data"]["algorithm"] == "bcrypt"
            assert hash_data["data"]["iterations"] == 10000
            assert "salt" in hash_data["data"]
            assert "hash" in hash_data["data"]
        
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
            
            if hasattr(aes_result, 'output'):
                aes_data = json.loads(aes_result.output)
            else:
                aes_data = aes_result
            
            assert aes_data["success"] is True
            assert aes_data["data"]["algorithm"] == "aes"
            assert aes_data["data"]["key_size"] == 256
            assert "key" in aes_data["data"]
            assert "iv" in aes_data["data"]
            
            # Verify key is base64 encoded and correct length
            key_bytes = base64.b64decode(aes_data["data"]["key"])
            assert len(key_bytes) == 32  # 256 bits = 32 bytes
            
            # Generate RSA key pair
            rsa_result = await injector.run('crypto', {
                "operation": "generate_key",
                "algorithm": "rsa",
                "key_size": 2048
            })
            
            if hasattr(rsa_result, 'output'):
                rsa_data = json.loads(rsa_result.output)
            else:
                rsa_data = rsa_result
            
            assert rsa_data["success"] is True
            assert rsa_data["data"]["algorithm"] == "rsa"
            assert rsa_data["data"]["key_size"] == 2048
            assert "private_key" in rsa_data["data"]
            assert "public_key" in rsa_data["data"]
        
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
            
            if hasattr(key_result, 'output'):
                key_data = json.loads(key_result.output)
            else:
                key_data = key_result
            
            encryption_key = key_data["data"]["key"]
            iv = key_data["data"]["iv"]
            
            # Encrypt data
            encrypt_result = await injector.run('crypto', {
                "operation": "encrypt",
                "algorithm": "aes",
                "data": test_data,
                "key": encryption_key,
                "iv": iv
            })
            
            if hasattr(encrypt_result, 'output'):
                encrypt_data = json.loads(encrypt_result.output)
            else:
                encrypt_data = encrypt_result
            
            assert encrypt_data["success"] is True
            assert "ciphertext" in encrypt_data["data"]
            ciphertext = encrypt_data["data"]["ciphertext"]
            
            # Decrypt data
            decrypt_result = await injector.run('crypto', {
                "operation": "decrypt",
                "algorithm": "aes",
                "data": ciphertext,
                "key": encryption_key,
                "iv": iv
            })
            
            if hasattr(decrypt_result, 'output'):
                decrypt_data = json.loads(decrypt_result.output)
            else:
                decrypt_data = decrypt_result
            
            assert decrypt_data["success"] is True
            assert decrypt_data["data"]["plaintext"] == test_data
        
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
            
            if hasattr(key_result, 'output'):
                key_data = json.loads(key_result.output)
            else:
                key_data = key_result
            
            private_key = key_data["data"]["private_key"]
            public_key = key_data["data"]["public_key"]
            
            # Sign data
            sign_result = await injector.run('crypto', {
                "operation": "sign",
                "algorithm": "rsa",
                "data": test_data,
                "private_key": private_key
            })
            
            if hasattr(sign_result, 'output'):
                sign_data = json.loads(sign_result.output)
            else:
                sign_data = sign_result
            
            assert sign_data["success"] is True
            signature = sign_data["data"]["signature"]
            
            # Verify signature
            verify_result = await injector.run('crypto', {
                "operation": "verify_signature",
                "algorithm": "rsa",
                "data": test_data,
                "signature": signature,
                "public_key": public_key
            })
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            assert verify_data["success"] is True
            assert verify_data["data"]["valid"] is True
            
            # Verify with wrong data
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_signature",
                "algorithm": "rsa",
                "data": "Wrong message",
                "signature": signature,
                "public_key": public_key
            })
            
            if hasattr(wrong_verify, 'output'):
                wrong_data = json.loads(wrong_verify.output)
            else:
                wrong_data = wrong_verify
            
            assert wrong_data["data"]["valid"] is False
        
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
            
            if hasattr(encode_result, 'output'):
                encode_data = json.loads(encode_result.output)
            else:
                encode_data = encode_result
            
            assert encode_data["success"] is True
            encoded = encode_data["data"]["encoded"]
            assert encode_data["data"]["original_length"] == len(test_data)
            
            # Decode from base64
            decode_result = await injector.run('crypto', {
                "operation": "decode_base64",
                "data": encoded
            })
            
            if hasattr(decode_result, 'output'):
                decode_data = json.loads(decode_result.output)
            else:
                decode_data = decode_result
            
            assert decode_data["success"] is True
            assert decode_data["data"]["decoded"] == test_data
        
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
            
            if hasattr(jwt_result, 'output'):
                jwt_data = json.loads(jwt_result.output)
            else:
                jwt_data = jwt_result
            
            assert jwt_data["success"] is True
            token = jwt_data["data"]["token"]
            assert len(token.split('.')) == 3  # JWT has 3 parts
            
            # Verify JWT
            verify_result = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": jwt_secret
            })
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            assert verify_data["success"] is True
            assert verify_data["data"]["valid"] is True
            assert verify_data["data"]["payload"]["user_id"] == "123"
            assert verify_data["data"]["payload"]["username"] == "testuser"
            
            # Verify with wrong secret
            wrong_verify = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": "wrong_secret"
            })
            
            if hasattr(wrong_verify, 'output'):
                wrong_data = json.loads(wrong_verify.output)
            else:
                wrong_data = wrong_verify
            
            assert wrong_data["success"] is True
            assert wrong_data["data"]["valid"] is False
        
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
            
            if hasattr(jwt_result, 'output'):
                jwt_data = json.loads(jwt_result.output)
            else:
                jwt_data = jwt_result
            
            token = jwt_data["data"]["token"]
            
            # Verify expired JWT
            verify_result = await injector.run('crypto', {
                "operation": "verify_jwt",
                "data": token,
                "secret": jwt_secret
            })
            
            if hasattr(verify_result, 'output'):
                verify_data = json.loads(verify_result.output)
            else:
                verify_data = verify_result
            
            assert verify_data["success"] is True
            assert verify_data["data"]["valid"] is False
            assert "expired" in verify_data["data"]["error"].lower()
        
        asyncio.run(run_test())
    
    def test_crypto_generate_salt(self):
        """Test salt generation."""
        
        async def run_test():
            injector = get_injector()
            
            # Generate salt
            salt_result = await injector.run('crypto', {
                "operation": "generate_salt"
            })
            
            if hasattr(salt_result, 'output'):
                salt_data = json.loads(salt_result.output)
            else:
                salt_data = salt_result
            
            assert salt_data["success"] is True
            salt = salt_data["data"]["salt"]
            assert salt_data["data"]["length"] == 16
            
            # Verify salt is base64 encoded
            salt_bytes = base64.b64decode(salt)
            assert len(salt_bytes) == 16
            
            # Generate another salt - should be different
            salt2_result = await injector.run('crypto', {
                "operation": "generate_salt"
            })
            
            if hasattr(salt2_result, 'output'):
                salt2_data = json.loads(salt2_result.output)
            else:
                salt2_data = salt2_result
            
            assert salt2_data["data"]["salt"] != salt
        
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
            
            if hasattr(hmac_result, 'output'):
                hmac_data = json.loads(hmac_result.output)
            else:
                hmac_data = hmac_result
            
            assert hmac_data["success"] is True
            assert hmac_data["data"]["algorithm"] == "sha256"
            hmac_value = hmac_data["data"]["hmac"]
            
            # Same data and key should produce same HMAC
            hmac2_result = await injector.run('crypto', {
                "operation": "hmac",
                "data": test_data,
                "key": test_key,
                "algorithm": "sha256"
            })
            
            if hasattr(hmac2_result, 'output'):
                hmac2_data = json.loads(hmac2_result.output)
            else:
                hmac2_data = hmac2_result
            
            assert hmac2_data["data"]["hmac"] == hmac_value
            
            # Different key should produce different HMAC
            hmac3_result = await injector.run('crypto', {
                "operation": "hmac",
                "data": test_data,
                "key": "different_key",
                "algorithm": "sha256"
            })
            
            if hasattr(hmac3_result, 'output'):
                hmac3_data = json.loads(hmac3_result.output)
            else:
                hmac3_data = hmac3_result
            
            assert hmac3_data["data"]["hmac"] != hmac_value
        
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
            
            # Test invalid JWT payload (empty)
            try:
                result = await injector.run('crypto', {
                    "operation": "generate_jwt",
                    "payload": {},
                    "secret": "test_secret",
                    "expires_in": 3600
                })
                assert False, "Expected ValueError to be raised for empty JWT payload"
            except ValueError as e:
                assert "JWT payload cannot be empty" in str(e)
            
            # Test empty JWT secret
            try:
                result = await injector.run('crypto', {
                    "operation": "generate_jwt",
                    "payload": {"test": "data"},
                    "secret": "",
                    "expires_in": 3600
                })
                assert False, "Expected ValueError to be raised for empty JWT secret"
            except ValueError as e:
                assert "JWT secret cannot be empty" in str(e)
        
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
            # Pydantic validation error is returned as output, not raised as exception
            if hasattr(result, 'output'):
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
            
            # Test empty HMAC key
            try:
                result = await injector.run('crypto', {
                    "operation": "hmac",
                    "data": "test data",
                    "key": "",
                    "algorithm": "sha256"
                })
                assert False, "Expected ValueError for empty HMAC key"
            except ValueError as e:
                assert "HMAC key cannot be empty" in str(e)
            
            # Test unsupported HMAC algorithm (Pydantic validation catches this)
            result = await injector.run('crypto', {
                "operation": "hmac",
                "data": "test data",
                "key": "test_key",
                "algorithm": "unsupported"
            })
            # Pydantic validation error is returned as output, not raised as exception
            if hasattr(result, 'output'):
                output = result.output
                assert "Error creating input model" in output
                assert "literal_error" in output
                assert "unsupported" in output
        
        asyncio.run(run_test())