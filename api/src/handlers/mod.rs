//! Request handlers for the knowledge API.

pub mod admin;
pub mod auth;
pub mod buffer;
// Credit-billing endpoints. Kept in their own modules so the feature
// can be removed (or replaced) without touching the rest of the API.
pub mod credits;
pub mod document_categories;
pub mod documents;
pub mod encode;
pub mod events;
pub mod exports;
pub mod favorite_docs;
pub mod favorites;
pub mod follows;
pub mod ingest;
pub mod mailer;
pub mod metadata;
pub mod personalities;
pub mod pipeline;
pub mod polar;
pub mod probe;
pub mod proxy;
pub mod rerank;
pub mod search;
pub mod secrets;
pub mod sponsorships;
pub mod storage;
pub mod tokens;
pub mod url_safety;
pub mod users;

pub use documents::*;
pub use encode::*;
pub use metadata::*;
pub use rerank::*;
pub use search::*;
