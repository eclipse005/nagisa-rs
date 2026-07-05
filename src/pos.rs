//! POS（词性）标注：对应 Python `nagisa.model.Model.encode_pt` + `tagger._postagging`。
//!
//! 流程：
//!   1. 由 words 构造 cids/wids/tids（含名词启发式）
//!   2. 对每个 word：vec_char = char-BiLSTM(char_emb[cids]) 取最后步；
//!      vec_word = WORD[wid]（wid==0 时零向量）；vec_tag = sum(POS[tid] or zero)
//!   3. pos-BiLSTM over concat(vec_word, vec_char, vec_tag)
//!   4. softmax(w_pos * h + b_pos) → argmax pid

use crate::weights::{LstmWeights, Weights};

/// 标准 BiLSTM 前向：输入 (N, in_dim) row-major，返回 (N, 2*h_dir) row-major，
/// 第 t 行 = [fwd_h[t]; bwd_h[t]]，与 dy.BiRNNBuilder(...).transduce 一致。
pub fn bilstm(xs: &[f32], in_dim: usize, fwd: &LstmWeights, bwd: &LstmWeights) -> Vec<f32> {
    let n = if in_dim == 0 { 0 } else { xs.len() / in_dim };
    let h_dir = fwd.b.len() / 4;
    if n == 0 {
        return Vec::new();
    }
    let fwd_out = crate::lstm::lstm_forward(xs, in_dim, fwd);
    // 反向
    let mut xs_rev = vec![0.0f32; xs.len()];
    for t in 0..n {
        let src = (n - 1 - t) * in_dim;
        let dst = t * in_dim;
        xs_rev[dst..dst + in_dim].copy_from_slice(&xs[src..src + in_dim]);
    }
    let bwd_rev = crate::lstm::lstm_forward(&xs_rev, in_dim, bwd);
    let dim_out = 2 * h_dir;
    let mut out = vec![0.0f32; n * dim_out];
    for t in 0..n {
        let dst = t * dim_out;
        out[dst..dst + h_dir].copy_from_slice(&fwd_out[t * h_dir..(t + 1) * h_dir]);
        let bsrc = (n - 1 - t) * h_dir;
        out[dst + h_dir..dst + 2 * h_dir].copy_from_slice(&bwd_rev[bsrc..bsrc + h_dir]);
    }
    out
}

fn row(table: &ndarray::Array2<f32>, id: u32, dim: usize) -> &[f32] {
    let s = table.as_slice_memory_order().expect("contiguous");
    &s[(id as usize) * dim..(id as usize + 1) * dim]
}

/// 对单个词：跑 char-BiLSTM，返回最后一步的 vec_char（dim = 2*char_h = 32）。
fn vec_char(cids: &[u32], w: &Weights, dim_uni: usize) -> Vec<f32> {
    let n = cids.len();
    let dim_out = 2 * (w.char_fwd.b.len() / 4);
    if n == 0 {
        // dy.transduce([]) 行为未定义；nagisa 词至少 1 字符，这里返回 0 向量兜底。
        return vec![0.0; dim_out];
    }
    let mut xs = vec![0.0f32; n * dim_uni];
    for (i, &cid) in cids.iter().enumerate() {
        let v = row(&w.uni_emb, cid, dim_uni);
        xs[i * dim_uni..(i + 1) * dim_uni].copy_from_slice(v);
    }
    let out = bilstm(&xs, dim_uni, &w.char_fwd, &w.char_bwd);
    out[(n - 1) * dim_out..n * dim_out].to_vec()
}

/// POS 标注主流程。返回每个词的 pid（pos id）。
///
/// - `cids_per_word`/`wids`/`tids`：与 Python `tagger._postagging` 构造的 X 一致
/// - `dim_word`/`dim_uni`/`dim_tagemb`：来自 hp
pub fn encode_pt(
    cids_per_word: &[Vec<u32>],
    wids: &[u32],
    tids: &[Vec<u32>],
    w: &Weights,
    dim_word: usize,
    dim_uni: usize,
    dim_tagemb: usize,
) -> Vec<usize> {
    encode_pt_inner(cids_per_word, wids, tids, w, dim_word, dim_uni, dim_tagemb).1
}

fn encode_pt_inner(
    cids_per_word: &[Vec<u32>],
    wids: &[u32],
    tids: &[Vec<u32>],
    w: &Weights,
    dim_word: usize,
    dim_uni: usize,
    dim_tagemb: usize,
) -> (Vec<f32>, Vec<usize>) {
    let n = cids_per_word.len();
    let vec_char_dim = 2 * (w.char_fwd.b.len() / 4); // 32
    let row_dim = dim_word + vec_char_dim + dim_tagemb; // 64
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut ipts = vec![0.0f32; n * row_dim];
    for i in 0..n {
        let mut p = i * row_dim;
        // vec_word
        if wids[i] != 0 {
            let v = row(&w.word_emb, wids[i], dim_word);
            ipts[p..p + dim_word].copy_from_slice(v);
        }
        p += dim_word;
        // vec_char
        let vc = vec_char(&cids_per_word[i], w, dim_uni);
        ipts[p..p + vec_char_dim].copy_from_slice(&vc);
        p += vec_char_dim;
        // vec_tag = esum(tid ? POS[tid] : zero)
        let mut acc = vec![0.0f32; dim_tagemb];
        for &tid in &tids[i] {
            if tid != 0 {
                let v = row(&w.pos_emb, tid, dim_tagemb);
                for k in 0..dim_tagemb {
                    acc[k] += v[k];
                }
            }
        }
        ipts[p..p + dim_tagemb].copy_from_slice(&acc);
    }
    // pos-BiLSTM
    let hidden = bilstm(&ipts, row_dim, &w.pos_fwd, &w.pos_bwd);
    let dim_hidden = 2 * (w.pos_fwd.b.len() / 4); // 100
    let n_pos = w.b_pos.len(); // 24
    let wpos_s = w.w_pos.as_slice_memory_order().unwrap();
    let mut pids = Vec::with_capacity(n);
    for i in 0..n {
        let h = &hidden[i * dim_hidden..(i + 1) * dim_hidden];
        // softmax 的 argmax 与 logits 的 argmax 一致，直接比较 logits。
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for tag in 0..n_pos {
            let wrow = &wpos_s[tag * dim_hidden..(tag + 1) * dim_hidden];
            let mut acc = w.b_pos[tag];
            for k in 0..dim_hidden {
                acc += wrow[k] * h[k];
            }
            if acc > best_v {
                best_v = acc;
                best = tag;
            }
        }
        pids.push(best);
    }
    (hidden, pids)
}
