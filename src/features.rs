//! Python `nagisa_utils.feature_extraction` 的等价实现。
//!
//! 输入：预处理+lower 后的文本（Python 端 `text = lower_text`）。
//! 输出：5 个长度均为 N 的序列：`[uids_cwin, bids_cwin, cids_cwin, start, end]`，
//! 其中前 3 个的内层是 `Vec<u32>`（窗口宽度 3），后 2 个的内层是 `Vec<u32>`
//! （任意长度，词典命中 id 列表）。

use unicode_normalization::UnicodeNormalization;

use crate::weights::Vocab;

pub struct Features {
    pub uids: Vec<[u32; 3]>,  // window=3，固定
    pub bids: Vec<[u32; 3]>,
    pub cids: Vec<[u32; 3]>,
    pub start: Vec<Vec<u32>>,
    pub end: Vec<Vec<u32>>,
}

/// 字符类型，对应 `get_chartype`。
/// 0=平假名 1=片假名 2=汉字 3=字母 4=数字 5=其它
fn chartype(c: char) -> u32 {
    if ('\u{3040}'..='\u{309F}').contains(&c) {
        return 0;
    }
    if ('\u{30A1}'..='\u{30FA}').contains(&c) {
        return 1;
    }
    if ('\u{4e00}'..='\u{9fa5}').contains(&c) {
        return 2;
    }
    // Python 端先对单字符做 NFKC，再判断字母/数字。
    let nfkcd: String = std::iter::once(c).nfkc().collect();
    for nc in nfkcd.chars() {
        if nc.is_ascii_alphabetic() {
            return 3;
        }
        if nc.is_ascii_digit() {
            return 4;
        }
    }
    5
}

fn conv_tokens_to_ids(words: &[String], idmap: &Vocab, oov: u32) -> Vec<u32> {
    words
        .iter()
        .map(|w| *idmap.get(w).unwrap_or(&oov))
        .collect()
}

fn cwin(l: &[u32], win: usize, pad: u32) -> Vec<Vec<u32>> {
    // win 应为奇数；half 两侧各 pad。
    debug_assert!(win % 2 == 1);
    let h = win / 2;
    let mut padded: Vec<u32> = vec![pad; h];
    padded.extend_from_slice(l);
    padded.extend(std::iter::repeat(pad).take(h));
    let mut out = Vec::with_capacity(l.len());
    for i in 0..l.len() {
        out.push(padded[i..i + win].to_vec());
    }
    out
}

fn get_bigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i == n - 1 {
            let mut s = String::with_capacity(chars[i].len_utf8() + 3);
            s.push(chars[i]);
            s.push_str("<E>");
            out.push(s);
        } else {
            let mut s = String::with_capacity(chars[i].len_utf8() + chars[i + 1].len_utf8());
            s.push(chars[i]);
            s.push(chars[i + 1]);
            out.push(s);
        }
    }
    out
}

pub struct VocabConfig {
    pub uni2id: Vocab,
    pub bi2id: Vocab,
    pub word2id: Vocab,
    pub window_size: usize,
}

/// 单字符 oov id（uni2id 的 'oov'，默认 0）。
pub fn oov_uni(uni2id: &Vocab) -> u32 {
    *uni2id.get("oov").unwrap_or(&0)
}
/// bi2id 的 'oov'，默认 0。
pub fn oov_bi(bi2id: &Vocab) -> u32 {
    *bi2id.get("oov").unwrap_or(&0)
}
/// word2id 的 'oov'（导出实测为 17）。
pub fn oov_word(word2id: &Vocab) -> u32 {
    *word2id.get("oov").unwrap_or(&17)
}
/// word2id 的 'pad'（导出实测为 1）。
pub fn pad_word(word2id: &Vocab) -> u32 {
    *word2id.get("pad").unwrap_or(&1)
}

pub fn feature_extraction(text: &str, cfg: &VocabConfig) -> Features {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let win = cfg.window_size;

    // unigrams（逐字符）
    let unigrams: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    let uni_ids = conv_tokens_to_ids(&unigrams, &cfg.uni2id, oov_uni(&cfg.uni2id));
    let bigrams = get_bigrams(text);
    let bi_ids = conv_tokens_to_ids(&bigrams, &cfg.bi2id, oov_bi(&cfg.bi2id));
    let cid_ids: Vec<u32> = chars.iter().map(|&c| chartype(c)).collect();

    // words_starting_at_i：j in i..min(i+8, n)，升序，最长匹配（即长串优先不强制——
    // Python 实际是按 j 升序收集所有命中，所以这里也收集全部命中，顺序一致）。
    let oov_w = oov_word(&cfg.word2id);
    let mut start = Vec::with_capacity(n);
    for i in 0..n {
        let mut hits: Vec<u32> = Vec::new();
        let upper = std::cmp::min(i + 8, n);
        for j in i..upper {
            let sub: String = chars[i..=j].iter().collect();
            if let Some(&id) = cfg.word2id.get(&sub) {
                hits.push(id);
            }
        }
        if hits.is_empty() {
            hits.push(oov_w);
        }
        start.push(hits);
    }

    // words_ending_at_i：Python 端先 `text = text[::-1]`，再做与 starting 相同的
    // 「正向」扫描，最后整体 `[::-1]` 翻转回来。等价于：对原位置 i，j 从 i 起向
    // 左走最多 8 步（降序），收集 substring[j..=i] 命中。这里直接用等价实现。
    let mut end = Vec::with_capacity(n);
    for i in 0..n {
        let mut hits: Vec<u32> = Vec::new();
        // j 范围：i, i-1, ..., max(0, i-7) 但 Python 的写法是 j in range(i, max(-1, i-8), -1)
        // 即 j 从 i 起递减，直到 > max(-1, i-8)。当 i>=8 时下界是 i-8（不含），即 j ∈ [i-7, i]。
        let lower = if i >= 8 { i as isize - 7 } else { 0 };
        let mut j = i as isize;
        while j >= lower {
            let jj = j as usize;
            let sub: String = chars[jj..=i].iter().collect();
            if let Some(&id) = cfg.word2id.get(&sub) {
                hits.push(id);
            }
            j -= 1;
        }
        if hits.is_empty() {
            hits.push(oov_w);
        }
        end.push(hits);
    }

    let pad_w = pad_word(&cfg.word2id);
    let pad_ctype = 6u32;

    let cw_u = cwin(&uni_ids, win, pad_w);
    let cw_b = cwin(&bi_ids, win, pad_w);
    let cw_c = cwin(&cid_ids, win, pad_ctype);

    // window_size 固定为 3（hp），把 Vec<u32> 转成 [u32;3] 方便后续索引。
    debug_assert_eq!(win, 3);
    let to_arr = |v: Vec<Vec<u32>>| -> Vec<[u32; 3]> {
        v.into_iter()
            .map(|w| [w[0], w[1], w[2]])
            .collect()
    };

    Features {
        uids: to_arr(cw_u),
        bids: to_arr(cw_b),
        cids: to_arr(cw_c),
        start,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg() -> VocabConfig {
        let mut uni = HashMap::new();
        uni.insert("oov".to_string(), 0);
        uni.insert("pad".to_string(), 1);
        uni.insert("あ".to_string(), 2);
        let mut bi = HashMap::new();
        bi.insert("oov".to_string(), 0);
        bi.insert("pad".to_string(), 1);
        bi.insert("あ<E>".to_string(), 5);
        let mut word = HashMap::new();
        word.insert("oov".to_string(), 17);
        word.insert("pad".to_string(), 1);
        word.insert("あ".to_string(), 3);
        VocabConfig {
            uni2id: uni,
            bi2id: bi,
            word2id: word,
            window_size: 3,
        }
    }

    #[test]
    fn chartype_basic() {
        assert_eq!(chartype('あ'), 0); // 平假
        assert_eq!(chartype('ア'), 1); // 片假
        assert_eq!(chartype('漢'), 2); // 汉字
        assert_eq!(chartype('A'), 3); // ASCII 字母
        assert_eq!(chartype('5'), 4); // 数字
        assert_eq!(chartype('。'), 5); // 其它
    }

    #[test]
    fn bigrams_end_symbol() {
        let b = get_bigrams("ab");
        assert_eq!(b, vec!["ab".to_string(), "b<E>".to_string()]);
    }

    #[test]
    fn feature_extraction_short() {
        let f = feature_extraction("あ", &cfg());
        assert_eq!(f.uids.len(), 1);
        // pad_word = 1
        assert_eq!(f.uids[0], [1, 2, 1]);
        assert_eq!(f.start[0], vec![3]);
    }
}
