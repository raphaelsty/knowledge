//! Escape user input that gets interpolated into a SQL `LIKE`/`ILIKE`
//! pattern.
//!
//! Without this, a query of `%` matches everything (DoS / enumeration)
//! and `_` matches arbitrary single chars. The `\` is the default
//! PostgreSQL `LIKE` escape character, so escaping `\` → `\\`,
//! `%` → `\%`, `_` → `\_` makes a literal pattern.
//!
//! Callers should still pass the escaped pattern via `.bind()` — this
//! is not a substitute for parameterised queries, it's an additional
//! layer that converts a wildcard injection into a literal match.

/// Escape PostgreSQL `LIKE` / `ILIKE` metacharacters (`\`, `%`, `_`).
pub fn escape_like_pattern(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if c == '\\' || c == '%' || c == '_' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_metachars_unchanged() {
        assert_eq!(escape_like_pattern("hello world"), "hello world");
    }

    #[test]
    fn percent_escaped() {
        assert_eq!(escape_like_pattern("100%"), "100\\%");
    }

    #[test]
    fn underscore_escaped() {
        assert_eq!(escape_like_pattern("user_id"), "user\\_id");
    }

    #[test]
    fn backslash_escaped() {
        assert_eq!(escape_like_pattern(r"a\b"), r"a\\b");
    }

    #[test]
    fn full_wildcard_collapses_to_literal() {
        assert_eq!(escape_like_pattern("%"), "\\%");
        assert_eq!(escape_like_pattern("%_%"), "\\%\\_\\%");
    }
}
