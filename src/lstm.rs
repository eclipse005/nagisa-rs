//! LSTM 前向、双向拼接、投影与 Viterbi/解码。
//!
//! 这是数学层。所有公式严格对应 README §3.3 与 §4：
//!   z = W x + U h + b
//!   i = sig(z[0:H]);      f = sig(z[H:2H] + 1.0)   ← forget_gate_bias = 1.0
//!   o = sig(z[2H:3H]);    g = tanh(z[3H:4H])
//!   c = f * c_prev + i * g
//!   h = o * tanh(c)
//!
//! 双向：先正向跑得到 o_f，再把输入反序跑得到 o_b_rev，最后整体反序得 o_b，
//! 拼接顺序 `[o_f; o_b]`（与 dynet BiRNNBuilder 一致，README 已验证）。

use crate::features::Features;
use crate::weights::{LstmWeights, Weights};

const FORGET_GATE_BIAS: f32 = 1.0;

/// 从 row-major 查找表中取出 id 对应的一行。
fn emb_row_fn(table: &ndarray::Array2<f32>, id: u32, dim: usize) -> &[f32] {
    let s = table.as_slice_memory_order().expect("contiguous");
    &s[(id as usize) * dim..(id as usize + 1) * dim]
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // 与 Python `1/(1+exp(-np.clip(x,-80,80)))` 一致：clip 防止溢出。
    let c = x.clamp(-80.0, 80.0);
    1.0 / (1.0 + (-c).exp())
}

/// 在单条样本上跑单向 LSTM。
/// `xs`：形状 (N, in_dim) row-major；返回 (N, H) row-major。
pub fn lstm_forward(xs: &[f32], in_dim: usize, lstm: &LstmWeights) -> Vec<f32> {
    let h_dim = lstm.b.len() / 4;
    let n = xs.len() / in_dim;
    let mut h_state = vec![0.0f32; h_dim];
    let mut c_state = vec![0.0f32; h_dim];
    let mut out = vec![0.0f32; n * h_dim];

    // 预分配 z 缓冲。
    let mut z = vec![0.0f32; 4 * h_dim];

    for t in 0..n {
        // z = W @ x  + U @ h_prev + b
        // W: (4H, in)，U: (4H, H)，row-major
        for r in 0..4 * h_dim {
            let mut acc = lstm.b[r];
            let wrow = &lstm.w.as_slice_memory_order().unwrap()[r * in_dim..(r + 1) * in_dim];
            let xrow = &xs[t * in_dim..(t + 1) * in_dim];
            for c in 0..in_dim {
                acc += wrow[c] * xrow[c];
            }
            let urow = &lstm.u.as_slice_memory_order().unwrap()[r * h_dim..(r + 1) * h_dim];
            for c in 0..h_dim {
                acc += urow[c] * h_state[c];
            }
            z[r] = acc;
        }

        for k in 0..h_dim {
            let i_t = sigmoid(z[k]);
            let f_t = sigmoid(z[h_dim + k] + FORGET_GATE_BIAS);
            let o_t = sigmoid(z[2 * h_dim + k]);
            let g_t = z[3 * h_dim + k].tanh();
            let c_new = f_t * c_state[k] + i_t * g_t;
            c_state[k] = c_new;
            h_state[k] = o_t * c_new.tanh();
        }

        out[t * h_dim..(t + 1) * h_dim].copy_from_slice(&h_state);
    }
    out
}

/// 从 `Features` 构造 LSTM 输入矩阵 (N, WS_INPUT_DIM) row-major。
///
/// 每个位置 i：
///   concat(uni_emb[uw].ravel, bi_emb[bw].ravel, ctype_emb[cw].ravel,
///          word_emb[start].sum(0), word_emb[end].sum(0))
pub fn build_inputs(fe: &Features, w: &Weights, hp: &crate::weights::HyperParams) -> Vec<f32> {
    let n = fe.uids.len();
    let d_uni = hp.dim_uni;
    let d_bi = hp.dim_bi;
    let d_ct = hp.dim_ctype;
    let d_word = hp.dim_word;
    let row = d_uni * 3 + d_bi * 3 + d_ct * 3 + d_word * 2;
    let mut out = vec![0.0f32; n * row];

    let emb_row = emb_row_fn;

    for i in 0..n {
        let mut p = i * row;
        // uni: 3 个查表向量拼接
        for k in 0..3 {
            let v = emb_row(&w.uni_emb, fe.uids[i][k], d_uni);
            out[p..p + d_uni].copy_from_slice(v);
            p += d_uni;
        }
        // bi
        for k in 0..3 {
            let v = emb_row(&w.bi_emb, fe.bids[i][k], d_bi);
            out[p..p + d_bi].copy_from_slice(v);
            p += d_bi;
        }
        // ctype
        for k in 0..3 {
            let v = emb_row(&w.ctype_emb, fe.cids[i][k], d_ct);
            out[p..p + d_ct].copy_from_slice(v);
            p += d_ct;
        }
        // start: sum(0)
        let mut acc_s = vec![0.0f32; d_word];
        for &id in &fe.start[i] {
            let v = emb_row(&w.word_emb, id, d_word);
            for j in 0..d_word {
                acc_s[j] += v[j];
            }
        }
        out[p..p + d_word].copy_from_slice(&acc_s);
        p += d_word;
        // end
        let mut acc_e = vec![0.0f32; d_word];
        for &id in &fe.end[i] {
            let v = emb_row(&w.word_emb, id, d_word);
            for j in 0..d_word {
                acc_e[j] += v[j];
            }
        }
        out[p..p + d_word].copy_from_slice(&acc_e);
    }
    out
}

/// 计算 observations：(N, 6) row-major。
/// obs = bilstm_concat @ w_ws.T + b_ws，其中 bilstm_concat = [fwd; bwd] (N, dim_hidden)
pub fn encode(fe: &Features, w: &Weights, hp: &crate::weights::HyperParams) -> Vec<f32> {
    let xs = build_inputs(fe, w, hp);
    let n = fe.uids.len();
    let in_dim = xs.len() / n.max(1);

    let fwd = lstm_forward(&xs, in_dim, &w.ws_fwd);
    // 反向：先把 xs 行序反序，跑 LSTM，再把输出行序反序。
    let mut xs_rev = vec![0.0f32; xs.len()];
    for t in 0..n {
        let src = (n - 1 - t) * in_dim;
        let dst = t * in_dim;
        xs_rev[dst..dst + in_dim].copy_from_slice(&xs[src..src + in_dim]);
    }
    let bwd_rev = lstm_forward(&xs_rev, in_dim, &w.ws_bwd);
    // bwd = reverse(bwd_rev) along time
    let h_dir = w.ws_fwd.b.len() / 4; // 50

    // big = concat([fwd_h, bwd_h]) -> (N, 2*h_dir) = (N, dim_hidden)
    let dim_hidden = 2 * h_dir;
    let mut big = vec![0.0f32; n * dim_hidden];
    for t in 0..n {
        let fsrc = t * h_dir;
        let bsrc = (n - 1 - t) * h_dir; // reverse
        let dst = t * dim_hidden;
        big[dst..dst + h_dir].copy_from_slice(&fwd[fsrc..fsrc + h_dir]);
        big[dst + h_dir..dst + 2 * h_dir].copy_from_slice(&bwd_rev[bsrc..bsrc + h_dir]);
    }

    // obs = big @ w_ws.T + b_ws  -> (N, 6)
    // w_ws: (6, dim_hidden) row-major；输出 6 维
    let n_tags = w.b_ws.len();
    let mut obs = vec![0.0f32; n * n_tags];
    let wws_s = w.w_ws.as_slice_memory_order().unwrap();
    for t in 0..n {
        let x = &big[t * dim_hidden..(t + 1) * dim_hidden];
        for tag in 0..n_tags {
            let row = &wws_s[tag * dim_hidden..(tag + 1) * dim_hidden];
            let mut acc = w.b_ws[tag];
            for k in 0..dim_hidden {
                acc += row[k] * x[k];
            }
            obs[t * n_tags + tag] = acc;
        }
    }
    obs
}

/// Viterbi 解码。严格对应 `nagisa_utils.np_viterbi`：
///   init T = [-1e10]*6 ; T[sp_s=4] = 0
///   for each obs: for nxt in 0..6: v = T + trans[nxt]; pick argmax; new T = v[argmax] + obs[nxt]
///   term = T + trans[sp_e=5]; pick argmax；回溯
pub fn viterbi(obs: &[f32], trans: &[f32], n_tags: usize) -> Vec<usize> {
    const SP_S: usize = 4;
    const SP_E: usize = 5;
    const NEG_INF: f32 = -1e10;

    let n = obs.len() / n_tags;
    let mut tcur = vec![NEG_INF; n_tags];
    tcur[SP_S] = 0.0;

    let mut backptrs: Vec<Vec<u8>> = Vec::with_capacity(n);

    for step in 0..n {
        let o = &obs[step * n_tags..(step + 1) * n_tags];
        let mut vvars = vec![NEG_INF; n_tags];
        let mut bptrs = vec![0u8; n_tags];
        for nxt in 0..n_tags {
            // v = tcur + trans[nxt][:]
            let trow = &trans[nxt * n_tags..(nxt + 1) * n_tags];
            let mut best = NEG_INF;
            let mut best_i: usize = 0;
            for prev in 0..n_tags {
                let val = tcur[prev] + trow[prev];
                if val > best {
                    best = val;
                    best_i = prev;
                }
            }
            bptrs[nxt] = best_i as u8;
            vvars[nxt] = best + o[nxt];
        }
        tcur = vvars;
        backptrs.push(bptrs);
    }

    // terminal: tcur + trans[sp_e][:]
    let terow = &trans[SP_E * n_tags..(SP_E + 1) * n_tags];
    let mut best = NEG_INF;
    let mut best_last: usize = 0;
    for prev in 0..n_tags {
        let val = tcur[prev] + terow[prev];
        if val > best {
            best = val;
            best_last = prev;
        }
    }

    let mut path = Vec::with_capacity(n);
    path.push(best_last);
    let mut cur = best_last;
    for b in backptrs.iter().rev() {
        cur = b[cur] as usize;
        path.push(cur);
    }
    // path 现在是 [last, ..., first]（含 n+1 项），去掉首项并反转
    // 注意 np_viterbi 的实现：best_path 先 push last；再 for reversed(backpointer): append bp[best]，
    // 最后 pop()（去掉最后一个，即 sp_s），再 reverse()。
    // 这里我们 push 顺序是 [last, prev_of_last, ..., bp_t0_choice]，长度 n+1。
    // np 的 pop() 移除末尾、reverse 得到长度 n 的序列。
    // 我们的 path 末尾元素 = 反向遍历完 backptrs 后最后访问的，即 t=0 时被选中的 prev（应是 sp_s）。
    path.pop();
    path.reverse();
    path
}

/// BMES 解码：3=S,2=E,0=B,1=I。对应 `segmenter_for_bmes`。
pub fn segment(chars: &[char], tags: &[usize]) -> Vec<String> {
    let mut words = Vec::new();
    let mut partial = String::new();
    for (ch, &t) in chars.iter().zip(tags.iter()) {
        match t {
            3 => {
                // SINGLE：直接收尾
                if !partial.is_empty() {
                    // 理论上不会发生（合法 BMES 序列中 S 前应是 E/S），
                    // 但稳健起见冲刷一下。
                    words.push(std::mem::take(&mut partial));
                }
                words.push(ch.to_string());
            }
            2 => {
                // END
                partial.push(*ch);
                words.push(std::mem::take(&mut partial));
            }
            _ => {
                // B(0) / I(1) / 其它：累积
                partial.push(*ch);
            }
        }
    }
    if !partial.is_empty() {
        words.push(partial);
    }
    words
}
