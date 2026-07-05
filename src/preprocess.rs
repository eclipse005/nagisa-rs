//! Python `nagisa_utils.preprocess(text)` 的等价实现。
//!
//! 顺序（必须严格一致）：
//!   1. `utf8rstrip` （Python `text.rstrip()` 去除所有空白）
//!   2. `unicodedata.normalize('NFKC', text)`
//!   3. `text.replace('İ', 'I')` （U+0130）
//!   4. `text.replace(' ', '　')` （ASCII 空格 → 全角空格 U+3000）
//!
//! 注意：`wakati` 内部还会做一次 `text.lower()`（见 tagger.py:70），那一步
//! 在 `Tagger::wakati` 里完成，不在此函数。

use unicode_normalization::{UnicodeNormalization, Recompositions};
use std::str::Chars;

/// Python `unicodedata.normalize('NFKC', text)` 的迭代器形式。
pub fn normalize_nfkc(s: &str) -> Recompositions<Chars<'_>> {
    s.nfkc()
}

/// Python `str.rstrip()` 在 Rust 中的等价：去除末尾所有 Unicode 空白字符。
fn py_rstrip(s: &str) -> &str {
    s.trim_end()
}

/// 完整的 preprocess（不含 lower）。
pub fn preprocess(text: &str) -> String {
    let stripped = py_rstrip(text);
    let normalized: String = stripped.nfkc().collect();
    let no_dot_i: String = normalized.replace('\u{0130}', "I");
    no_dot_i.replace(' ', "\u{3000}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfkc_fullwidth_to_halfwidth() {
        // 全角 → 半角是 NFKC 的常见效果。
        let s = preprocess("ＡＢＣ");
        assert_eq!(s, "ABC");
    }

    #[test]
    fn space_to_fullwidth() {
        assert_eq!(preprocess("a b"), "a\u{3000}b");
    }

    #[test]
    fn dotless_i_replaced() {
        assert_eq!(preprocess("İ"), "I");
    }

    #[test]
    fn rstrip_works() {
        assert_eq!(preprocess("foo   "), "foo");
        assert_eq!(preprocess("foo\t\n"), "foo");
    }
}
