//! 加载 safetensors 权重与 hp.json / 词汇表，并封装为强类型结构。
//!
//! 对应 Python 端 `safetensors.numpy.load_file` 与 `json.load`。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::{Array1, Array2};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

/// 所有 WS（词切分）推理所需的权重。
///
/// 名称沿用 Python 端 `weights.safetensors` 的导出键。
pub struct Weights {
    // 查找表
    pub uni_emb: Array2<f32>,   // (3090, 32)
    pub bi_emb: Array2<f32>,    // (82114, 16)
    pub ctype_emb: Array2<f32>, // (7, 8)
    pub word_emb: Array2<f32>,  // (59260, 16)

    // 双向 LSTM（WS）
    pub ws_fwd: LstmWeights,
    pub ws_bwd: LstmWeights,

    // 输出投影 + CRF 转移
    pub w_ws: Array2<f32>,  // (6, 100)
    pub b_ws: Array1<f32>,  // (6,)
    pub trans: Array2<f32>, // (6, 6)

    // POS 标注所需权重
    pub pos_emb: Array2<f32>,  // (24, 16)
    pub char_fwd: LstmWeights, // (4*16=64, 32) / (64,16) / (64,)
    pub char_bwd: LstmWeights,
    pub pos_fwd: LstmWeights, // (4*50=200, 64) / (200,50) / (200,)
    pub pos_bwd: LstmWeights,
    pub w_pos: Array2<f32>, // (24, 100)
    pub b_pos: Array1<f32>, // (24,)
}

pub struct LstmWeights {
    pub w: Array2<f32>, // (4H, in)  -> (200, 200) 或方向内 (200, 200)
    pub u: Array2<f32>, // (4H, H)   -> (200, 50)
    pub b: Array1<f32>, // (4H,)     -> (200,)
}

#[derive(Debug, Deserialize)]
pub struct HyperParams {
    pub window_size: usize,
    pub dim_uni: usize,
    pub dim_bi: usize,
    pub dim_ctype: usize,
    pub dim_word: usize,
    pub dim_tagemb: usize,
    pub dim_hidden: usize,
    pub layers: usize,
}

// serde 默认会按字段名匹配；我们手动实现以容忍 JSON 用 SCREAMING_SNAKE。
// 见 `load_hp` 中的字段重命名。
#[derive(Debug, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
struct HpRaw {
    window_size: usize,
    dim_uni: usize,
    dim_bi: usize,
    dim_ctype: usize,
    dim_word: usize,
    dim_tagemb: usize,
    dim_hidden: usize,
    layers: usize,
}

pub fn load_hp_from_str(txt: &str) -> Result<HyperParams> {
    let raw: HpRaw = serde_json::from_str(txt).context("parse hp.json")?;
    Ok(HyperParams {
        window_size: raw.window_size,
        dim_uni: raw.dim_uni,
        dim_bi: raw.dim_bi,
        dim_ctype: raw.dim_ctype,
        dim_word: raw.dim_word,
        dim_tagemb: raw.dim_tagemb,
        dim_hidden: raw.dim_hidden,
        layers: raw.layers,
    })
}

pub fn load_hp(path: &Path) -> Result<HyperParams> {
    let txt =
        std::fs::read_to_string(path).with_context(|| format!("read hp {}", path.display()))?;
    load_hp_from_str(&txt)
}

/// 词汇表：String → id。
pub type Vocab = HashMap<String, u32>;

/// 加载形如 `{"<name>": { ... }}` 的嵌套 JSON 词汇表，取出内层 map。
pub fn load_vocab_from_str(txt: &str, key: &str) -> Result<Vocab> {
    let val: serde_json::Value = serde_json::from_str(txt).context("parse vocab json")?;
    let inner = val
        .get(key)
        .ok_or_else(|| anyhow!("vocab missing top-level key {key}"))?
        .as_object()
        .ok_or_else(|| anyhow!("vocab inner is not an object"))?;
    let mut out = HashMap::with_capacity(inner.len());
    for (k, v) in inner {
        let id = v
            .as_i64()
            .ok_or_else(|| anyhow!("vocab: id for {k:?} not integer"))? as u32;
        out.insert(k.clone(), id);
    }
    Ok(out)
}

/// 加载形如 `{"<name>": { ... }}` 的嵌套 JSON 词汇表文件，取出内层 map。
pub fn load_vocab(path: &Path, key: &str) -> Result<Vocab> {
    let txt =
        std::fs::read_to_string(path).with_context(|| format!("read vocab {}", path.display()))?;
    load_vocab_from_str(&txt, key)
        .with_context(|| format!("parse vocab {}", path.display()))
}

/// 加载 `{"<key>": {word: [pos_ids...]}}` 形式的 word→postags 映射。
pub fn load_word2postags_from_str(txt: &str, key: &str) -> Result<HashMap<String, Vec<u32>>> {
    let val: serde_json::Value = serde_json::from_str(txt).context("parse word2postags json")?;
    let inner = val
        .get(key)
        .ok_or_else(|| anyhow!("missing key {key}"))?
        .as_object()
        .ok_or_else(|| anyhow!("{key} not an object"))?;
    let mut out = HashMap::with_capacity(inner.len());
    for (word, v) in inner {
        let arr = v
            .as_array()
            .ok_or_else(|| anyhow!("word2postags[{word:?}] not array"))?;
        let ids: Vec<u32> = arr
            .iter()
            .map(|x| {
                x.as_i64()
                    .ok_or_else(|| anyhow!("word2postags[{word:?}] entry not int"))
                    .map(|i| i as u32)
            })
            .collect::<Result<_>>()?;
        out.insert(word.clone(), ids);
    }
    Ok(out)
}

/// 加载 `{"<key>": {word: [pos_ids...]}}` 形式的 word→postags 映射。
pub fn load_word2postags(path: &Path, key: &str) -> Result<HashMap<String, Vec<u32>>> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    load_word2postags_from_str(&txt, key)
        .with_context(|| format!("parse {}", path.display()))
}

fn view_f32<'a>(st: &'a SafeTensors<'a>, name: &str) -> Result<(&'a [u8], Vec<usize>)> {
    let t = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e}"))?;
    if t.dtype() != Dtype::F32 {
        return Err(anyhow!("tensor {name} expected F32, got {:?}", t.dtype()));
    }
    Ok((t.data(), t.shape().to_vec()))
}

fn to_array1(st: &SafeTensors<'_>, name: &str) -> Result<Array1<f32>> {
    let (data, shape) = view_f32(st, name)?;
    if shape.len() != 1 {
        return Err(anyhow!("{name}: expected 1-D, got {:?}", shape));
    }
    let n = shape[0];
    let mut out = Vec::with_capacity(n);
    let bytes = data
        .get(0..n * 4)
        .ok_or_else(|| anyhow!("{name}: short data"))?;
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Array1::from_vec(out))
}

fn to_array2(st: &SafeTensors<'_>, name: &str) -> Result<Array2<f32>> {
    let (data, shape) = view_f32(st, name)?;
    if shape.len() != 2 {
        return Err(anyhow!("{name}: expected 2-D, got {:?}", shape));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = Vec::with_capacity(rows * cols);
    let want = rows * cols * 4;
    let bytes = data
        .get(0..want)
        .ok_or_else(|| anyhow!("{name}: short data"))?;
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Array2::from_shape_vec((rows, cols), out)?)
}

fn lstm(st: &SafeTensors<'_>, prefix: &str) -> Result<LstmWeights> {
    Ok(LstmWeights {
        w: to_array2(st, &format!("{prefix}.W"))?,
        u: to_array2(st, &format!("{prefix}.U"))?,
        b: to_array1(st, &format!("{prefix}.b"))?,
    })
}

pub fn load_weights_from_bytes(raw: &[u8]) -> Result<Weights> {
    let st = SafeTensors::deserialize(raw).context("deserialize safetensors")?;
    Ok(Weights {
        uni_emb: to_array2(&st, "uni_emb")?,
        bi_emb: to_array2(&st, "bi_emb")?,
        ctype_emb: to_array2(&st, "ctype_emb")?,
        word_emb: to_array2(&st, "word_emb")?,
        ws_fwd: lstm(&st, "ws_fwd_lstm")?,
        ws_bwd: lstm(&st, "ws_bwd_lstm")?,
        w_ws: to_array2(&st, "w_ws")?,
        b_ws: to_array1(&st, "b_ws")?,
        trans: to_array2(&st, "trans")?,
        pos_emb: to_array2(&st, "pos_emb")?,
        char_fwd: lstm(&st, "char_fwd_lstm")?,
        char_bwd: lstm(&st, "char_bwd_lstm")?,
        pos_fwd: lstm(&st, "pos_fwd_lstm")?,
        pos_bwd: lstm(&st, "pos_bwd_lstm")?,
        w_pos: to_array2(&st, "w_pos")?,
        b_pos: to_array1(&st, "b_pos")?,
    })
}

pub fn load_weights(path: &Path) -> Result<Weights> {
    let raw = std::fs::read(path).with_context(|| format!("read weights {}", path.display()))?;
    load_weights_from_bytes(&raw)
}
