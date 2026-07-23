//! nagisa-rs：DyNet 双向 LSTM-CRF 日语分词器的纯 Rust 移植。
//!
//! 对外暴露 [`Tagger`]：
//! - [`Tagger::wakati`] 与 Python `nagisa.wakati(text)` 逐字匹配
//! - [`Tagger::tagging`] 与 Python `nagisa.tagging(text).words/.postags` 逐字匹配
//!
//! # 例
//!
//! ```no_run
//! use nagisa_rs::Tagger;
//! // Prefer the crate-embedded model (no external files needed):
//! let tagger = Tagger::embedded().unwrap();
//! // Or load from a directory of weights/dicts:
//! // let tagger = Tagger::new("models").unwrap();
//! let words = tagger.wakati("本日は晴天なり。");
//! let tagged = tagger.tagging("本日は晴天なり。");
//! ```

pub mod features;
pub mod lstm;
pub mod pos;
pub mod preprocess;
pub mod weights;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};

pub use features::{VocabConfig, feature_extraction};
pub use lstm::{encode, segment, viterbi};
pub use preprocess::preprocess;
pub use weights::{
    HyperParams, Vocab, Weights, load_hp, load_hp_from_str, load_vocab, load_vocab_from_str,
    load_weights, load_weights_from_bytes, load_word2postags, load_word2postags_from_str,
};

/// 词性标注结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaggingResult {
    pub words: Vec<String>,
    pub postags: Vec<String>,
}

pub struct Tagger {
    hp: HyperParams,
    cfg: VocabConfig,
    weights: Weights,
    trans_flat: Vec<f32>,
    // POS 相关
    pos2id: HashMap<String, u32>,
    id2pos: HashMap<u32, String>,
    word2postags: HashMap<String, Vec<u32>>,
    use_noun_heuristic: bool,
}

impl Tagger {
    /// Load the model that ships with this crate (`models/` via `include_*!`).
    ///
    /// No external files or download are required. Preferred for app embedding.
    pub fn embedded() -> Result<Self> {
        // Compile-time bundle of the seven files required for inference.
        // ~25 MB total; kept in the binary so consumers never need models/.
        let hp = load_hp_from_str(include_str!("../models/hp.json")).context("embedded hp")?;
        let weights = load_weights_from_bytes(include_bytes!("../models/weights.safetensors"))
            .context("embedded weights")?;
        let uni2id = load_vocab_from_str(include_str!("../models/uni2id.json"), "uni2id")
            .context("embedded uni2id")?;
        let bi2id = load_vocab_from_str(include_str!("../models/bi2id.json"), "bi2id")
            .context("embedded bi2id")?;
        let word2id = load_vocab_from_str(include_str!("../models/word2id.json"), "word2id")
            .context("embedded word2id")?;
        let pos2id_raw = load_vocab_from_str(include_str!("../models/pos2id.json"), "pos2id")
            .context("embedded pos2id")?;
        let word2postags =
            load_word2postags_from_str(include_str!("../models/word2postags.json"), "word2postags")
                .context("embedded word2postags")?;
        Ok(Self::from_parts(
            hp,
            weights,
            uni2id,
            bi2id,
            word2id,
            pos2id_raw,
            word2postags,
        ))
    }

    /// 从目录加载模型。期望目录包含：`hp.json`、`weights.safetensors`、
    /// `uni2id.json`、`bi2id.json`、`word2id.json`、`pos2id.json`、`word2postags.json`。
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        let hp = load_hp(&dir.join("hp.json"))?;
        let weights = load_weights(&dir.join("weights.safetensors"))?;
        let uni2id = load_vocab(&dir.join("uni2id.json"), "uni2id")?;
        let bi2id = load_vocab(&dir.join("bi2id.json"), "bi2id")?;
        let word2id = load_vocab(&dir.join("word2id.json"), "word2id")?;
        let pos2id_raw = load_vocab(&dir.join("pos2id.json"), "pos2id")?;
        let word2postags = load_word2postags(&dir.join("word2postags.json"), "word2postags")?;
        Ok(Self::from_parts(
            hp,
            weights,
            uni2id,
            bi2id,
            word2id,
            pos2id_raw,
            word2postags,
        ))
    }

    fn from_parts(
        hp: HyperParams,
        weights: Weights,
        uni2id: Vocab,
        bi2id: Vocab,
        word2id: Vocab,
        pos2id_raw: Vocab,
        word2postags: HashMap<String, Vec<u32>>,
    ) -> Self {
        let cfg = VocabConfig {
            uni2id,
            bi2id,
            word2id,
            window_size: hp.window_size,
        };
        let trans_flat = weights.trans.as_slice_memory_order().unwrap().to_vec();
        // pos2id.json 的值是 pos 字符串到 id（已确认）。
        let pos2id: HashMap<String, u32> =
            pos2id_raw.iter().map(|(k, &v)| (k.clone(), v)).collect();
        let id2pos: HashMap<u32, String> = pos2id.iter().map(|(k, &v)| (v, k.clone())).collect();
        let use_noun_heuristic = pos2id.contains_key("名詞");

        Self {
            hp,
            cfg,
            weights,
            trans_flat,
            pos2id,
            id2pos,
            word2postags,
            use_noun_heuristic,
        }
    }

    /// 词切分。等价于 `nagisa.wakati(text)`（不带 single_word_list）。
    pub fn wakati(&self, text: &str) -> Vec<String> {
        // 1. preprocess（rstrip → NFKC → İ→I → 空格→全角）
        let pre = preprocess(text);
        // 空输入：Python nagisa 对空串/纯空白返回 []。encode/viterbi 也无法处理 N=0，
        // 在此短路。
        if pre.is_empty() {
            return Vec::new();
        }
        // 2. lower （Python tagger.py:70 `lower_text = text.lower()`）
        let lower: String = pre.to_lowercase();
        // 3. feature extraction
        let fe = feature_extraction(&lower, &self.cfg);
        // 4. encode (BiLSTM + projection)
        let n_tags = self.weights.b_ws.len();
        let obs = encode(&fe, &self.weights, &self.hp);
        // 5. viterbi
        let tags = viterbi(&obs, &self.trans_flat, n_tags);
        // 6. segment（用未 lower 的 pre 文本，与 Python tagger.py:109 行为一致）
        let chars: Vec<char> = pre.chars().collect();
        segment(&chars, &tags)
    }

    /// 词性标注。等价于 `nagisa.tagging(text).words/.postags`。
    pub fn tagging(&self, text: &str) -> TaggingResult {
        let words = self.wakati(text);
        let postags = self.postag_words(&words);
        TaggingResult { words, postags }
    }

    /// 对已分好词的列表做 POS 标注。等价于 `nagisa._postagging(words)` +
    /// `tagger.decode` 的预处理（preprocess 每个词）。
    ///
    /// 注意 Python 端对每个词做 preprocess：空格/全角空格用 `preprocess_without_rstrip`，
    /// 其余用 `preprocess`（含 rstrip）。
    pub fn postag_words(&self, words: &[String]) -> Vec<String> {
        if words.is_empty() {
            return Vec::new();
        }
        // 1. preprocess 每个词
        let pp_words: Vec<String> = words
            .iter()
            .map(|w| {
                if w == " " || w == "\u{3000}" {
                    // preprocess_without_rstrip：只 NFKC + İ→I + 空格→全角（不 rstrip）
                    let n: String = crate::preprocess::normalize_nfkc(w).collect();
                    let n = n.replace('\u{0130}', "I");
                    n.replace(' ', "\u{3000}")
                } else {
                    preprocess(w)
                }
            })
            .collect();

        // 2. wids
        let oov_word = *self.cfg.word2id.get("oov").unwrap_or(&17);
        let wids: Vec<u32> = pp_words
            .iter()
            .map(|w| *self.cfg.word2id.get(w).unwrap_or(&oov_word))
            .collect();
        // 3. cids（每词逐字符）
        let oov_uni = *self.cfg.uni2id.get("oov").unwrap_or(&0);
        let cids: Vec<Vec<u32>> = pp_words
            .iter()
            .map(|w| {
                w.chars()
                    .map(|c| {
                        let s = c.to_string();
                        *self.cfg.uni2id.get(&s).unwrap_or(&oov_uni)
                    })
                    .collect()
            })
            .collect();
        // 4. tids（含名词启发式）
        let noun_id = self.pos2id.get("名詞").copied();
        let tids: Vec<Vec<u32>> = pp_words
            .iter()
            .map(|w| {
                let mut s: std::collections::HashSet<u32> = self
                    .word2postags
                    .get(w)
                    .map(|v| v.iter().copied().collect())
                    .unwrap_or_else(|| [0u32].into_iter().collect());
                if self.use_noun_heuristic {
                    if word_is_alnum(w) {
                        s.remove(&0);
                        if let Some(nid) = noun_id {
                            s.insert(nid);
                        }
                    }
                }
                s.into_iter().collect()
            })
            .collect();

        // 5. encode_pt → pids
        let pids = pos::encode_pt(
            &cids,
            &wids,
            &tids,
            &self.weights,
            self.hp.dim_word,
            self.hp.dim_uni,
            self.hp.dim_tagemb,
        );
        // 6. 映射回 pos 字符串
        pids.iter()
            .map(|&p| {
                self.id2pos
                    .get(&(p as u32))
                    .cloned()
                    .unwrap_or_else(|| "oov".to_string())
            })
            .collect()
    }
}

/// Python `str.isalnum()` 等价：任一字符是 Unicode 字母/数字即 True。
/// 注意：与 `is_alphanumeric`（Rust char 方法）一致；Python 还包含其它 Number 类别，
/// 但 `_postagging` 实际命中场景主要看 ASCII/汉字/假名，差异可忽略。
fn word_is_alnum(w: &str) -> bool {
    w.chars().any(|c| c.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> std::path::PathBuf {
        // 相对于 crate 根目录（Cargo.toml 所在处）定位 models/。
        let manifest = env!("CARGO_MANIFEST_DIR");
        std::path::PathBuf::from(manifest).join("models")
    }

    fn assert_wakati(t: &Tagger, text: &str, expected: &[&str]) {
        let got = t.wakati(text);
        let exp: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(got, exp, "input: {text:?}");
    }

    #[test]
    fn embedded_matches_directory_load() {
        let from_dir = Tagger::new(model_dir()).expect("load model from dir");
        let from_embed = Tagger::embedded().expect("load embedded model");
        let samples = [
            "シンデレラライライゼロ",
            "女子アナの仕事に耐える。辛抱大工です。",
            "本日は島根県にある有名な人気ラーメン店にやってきました。",
            "AIの進化が凄まじい。",
        ];
        for text in samples {
            assert_eq!(
                from_dir.wakati(text),
                from_embed.wakati(text),
                "wakati mismatch for {text:?}"
            );
            let a = from_dir.tagging(text);
            let b = from_embed.tagging(text);
            assert_eq!(a.words, b.words, "tagging words mismatch for {text:?}");
            assert_eq!(
                a.postags, b.postags,
                "tagging postags mismatch for {text:?}"
            );
        }
    }

    #[test]
    fn ref1_short_kana() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_wakati(&t, "シンデレライライゼロ", &["シンデレ", "ライライゼロ"]);
    }

    #[test]
    fn ref2_long_kana() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_wakati(
            &t,
            "クラブリベリオンシンデレライライゼロ",
            &["クラブリベリオン", "シンデレ", "ライライゼロ"],
        );
    }

    #[test]
    fn ref3_mixed() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_wakati(
            &t,
            "女子アナの仕事に耐える。辛抱大工です。",
            &[
                "女子",
                "アナ",
                "の",
                "仕事",
                "に",
                "耐える",
                "。",
                "辛抱",
                "大工",
                "です",
                "。",
            ],
        );
    }

    #[test]
    fn ref4_kanji_long() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_wakati(
            &t,
            "本日は島根県にある有名な人気ラーメン店にやってきました。",
            &[
                "本日",
                "は",
                "島根",
                "県",
                "に",
                "ある",
                "有名",
                "な",
                "人気",
                "ラーメン",
                "店",
                "に",
                "やっ",
                "て",
                "き",
                "まし",
                "た",
                "。",
            ],
        );
    }

    // ---- POS 标注（tagging）参考测试 ----
    // 期望值与 Python nagisa.tagging(text).words / .postags 逐字一致。
    fn assert_tagging(t: &Tagger, text: &str, words: &[&str], postags: &[&str]) {
        let r = t.tagging(text);
        let exp_w: Vec<String> = words.iter().map(|s| s.to_string()).collect();
        let exp_p: Vec<String> = postags.iter().map(|s| s.to_string()).collect();
        assert_eq!(r.words, exp_w, "words mismatch input: {text:?}");
        assert_eq!(r.postags, exp_p, "postags mismatch input: {text:?}");
    }

    #[test]
    fn tagging_ref1_long_kanji() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_tagging(
            &t,
            "本日は島根県にある有名な人気ラーメン店にやってきました。",
            &[
                "本日",
                "は",
                "島根",
                "県",
                "に",
                "ある",
                "有名",
                "な",
                "人気",
                "ラーメン",
                "店",
                "に",
                "やっ",
                "て",
                "き",
                "まし",
                "た",
                "。",
            ],
            &[
                "名詞",
                "助詞",
                "名詞",
                "名詞",
                "助詞",
                "動詞",
                "形状詞",
                "助動詞",
                "名詞",
                "名詞",
                "接尾辞",
                "助詞",
                "動詞",
                "助詞",
                "動詞",
                "助動詞",
                "助動詞",
                "補助記号",
            ],
        );
    }

    #[test]
    fn tagging_ref2_ascii_mixed() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_tagging(
            &t,
            "AIの進化が凄まじい。",
            &["AI", "の", "進化", "が", "凄まじい", "。"],
            &["名詞", "助詞", "名詞", "助詞", "形容詞", "補助記号"],
        );
    }

    #[test]
    fn tagging_ref3_kana_only() {
        let t = Tagger::new(model_dir()).expect("load model");
        assert_tagging(
            &t,
            "シンデレライライゼロ",
            &["シンデレ", "ライライゼロ"],
            &["名詞", "名詞"],
        );
    }

    #[test]
    fn tagging_empty_returns_empty() {
        let t = Tagger::new(model_dir()).expect("load model");
        let r = t.tagging("");
        assert!(r.words.is_empty());
        assert!(r.postags.is_empty());
    }

    #[test]
    fn tagging_postag_words_direct() {
        // postag_words 直接对词列表做标注（与 nagisa.postagging(words) 一致）。
        let t = Tagger::new(model_dir()).expect("load model");
        let words: Vec<String> = ["東京", "は", "日本", "の", "首都", "です", "。"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let postags = t.postag_words(&words);
        assert_eq!(
            postags,
            vec!["名詞", "助詞", "名詞", "助詞", "名詞", "助動詞", "補助記号"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
    }
}
