"""AES-256-GCM decryption for stored third-party credentials.

Matches the Rust encryption scheme in ``api/src/handlers/secrets.rs``:
    wire format = base64(nonce || ciphertext || tag)
    nonce       = 12 random bytes
    key         = 32 bytes from HN_ENCRYPTION_KEY (hex-encoded)

Rotating HN_ENCRYPTION_KEY invalidates every stored secret — users
will see the scraper log "HN auth unavailable" until they re-enter
their password via the profile form.
"""

from __future__ import annotations

import base64
import os

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover — cryptography may not be installed
    AESGCM = None  # type: ignore[assignment]


def decrypt(blob_b64: str) -> str | None:
    """Return the plaintext or ``None`` when decryption is impossible.

    Returns ``None`` for every failure mode (missing env, bad base64,
    auth-tag mismatch, cryptography lib absent). Callers treat that as
    "no credentials on file" rather than crashing the pipeline.
    """
    if AESGCM is None:
        return None
    key_hex = os.environ.get("HN_ENCRYPTION_KEY", "").strip()
    if len(key_hex) != 64:
        return None
    try:
        key = bytes.fromhex(key_hex)
    except ValueError:
        return None
    try:
        blob = base64.b64decode(blob_b64.strip())
    except Exception:
        return None
    if len(blob) <= 12:
        return None
    nonce, ct = blob[:12], blob[12:]
    try:
        pt = AESGCM(key).decrypt(nonce, ct, None)
    except Exception:
        return None
    try:
        return pt.decode("utf-8")
    except UnicodeDecodeError:
        return None
