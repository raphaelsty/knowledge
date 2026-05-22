//! Symmetric encryption for third-party credentials.
//!
//! Used to protect values we *must* keep retrievable in plaintext for the
//! pipeline to act on behalf of the user (e.g. the HackerNews password —
//! HN has no OAuth, so there's no token-for-password substitute).
//!
//! Scheme: AES-256-GCM with a 256-bit key read from the
//! ``HN_ENCRYPTION_KEY`` env var (hex-encoded, 64 chars). A fresh 12-byte
//! nonce is drawn per encryption; the wire format is
//! ``base64(nonce || ciphertext || tag)``.
//!
//! Rotating the key invalidates every stored secret, which is a
//! feature — logging in again re-encrypts under the new key. Losing the
//! key means the stored passwords are irretrievable (equally a feature).

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rand::RngCore;

const NONCE_LEN: usize = 12;

fn cipher_from_env() -> Option<Aes256Gcm> {
    let hex = std::env::var("HN_ENCRYPTION_KEY").ok()?;
    let key = hex::decode(hex.trim()).ok()?;
    if key.len() != 32 {
        return None;
    }
    Aes256Gcm::new_from_slice(&key).ok()
}

/// Encrypt a UTF-8 secret. Returns the base64 wire-format string.
/// `None` when `HN_ENCRYPTION_KEY` is not set or invalid — callers treat
/// that as "refuse to persist" rather than silently storing plaintext.
pub fn encrypt(plaintext: &str) -> Option<String> {
    let cipher = cipher_from_env()?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher.encrypt(nonce, plaintext.as_bytes()).ok()?;
    let mut blob = Vec::with_capacity(NONCE_LEN + ct.len());
    blob.extend_from_slice(&nonce_bytes);
    blob.extend_from_slice(&ct);
    Some(STANDARD.encode(&blob))
}

/// Not currently called by the API (the pipeline does decryption on
/// the Python side), but included for completeness and future use.
#[allow(dead_code)]
pub fn decrypt(blob_b64: &str) -> Option<String> {
    let cipher = cipher_from_env()?;
    let blob = STANDARD.decode(blob_b64.trim()).ok()?;
    if blob.len() <= NONCE_LEN {
        return None;
    }
    let (nonce_bytes, ct) = blob.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);
    let pt = cipher.decrypt(nonce, ct).ok()?;
    String::from_utf8(pt).ok()
}
