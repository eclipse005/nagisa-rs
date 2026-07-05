# nagisa-rs 移植任务（交接文档）

> 本文件替代原有的规划文档。当前进度：**Rust 端口已完成**——`cargo test` 全绿（16/16），`wakati`（词切分）与 `tagging`（POS 词性标注）均与 Python `nagisa` 逐字一致。各关键节点（vec_char / pos-BiLSTM hidden / softmax）对照 dynet 抽样 maxabs ≤ 6.5e-7（float32 噪声）。额外交叉验证：18 条文本（含空串/纯空白/数字/ASCII/引号/长句）全部匹配。接手者请直接阅读 `src/` 与本文档 §3/§4 的公式锁定，切勿重复已完成的工作。

## 1. 任务目标

将 `nagisa`（DyNet 双向 LSTM-CRF 日语分词器）移植为纯 Rust 库 `nagisa-rs`，输出与 Python 版 `nagisa.wakati(text)` / `nagisa.tagging(text).words` 逐字匹配。

**必须匹配的 4 条参考文本（当前已全匹配 3/4，第 2 条因 LSTM 多步误差暂时偏移）：**

```text
シンデレライライゼロ
-> ["シンデレ", "ライライゼロ"]

クラブリベリオンシンデレライライゼロ
-> ["クラブリベリオン", "シンデレ", "ライライゼロ"]

女子アナの仕事に耐える。辛抱大工です。
-> ["女子", "アナ", "の", "仕事", "に", "耐える", "。", "辛抱", "大工", "です", "。"]

本日は島根県にある有名な人気ラーメン店にやってきました。
-> ["本日", "は", "島根", "県", "に", "ある", "有名", "な", "人気", "ラーメン", "店", "に", "やっ", "て", "き", "まし", "た", "。"]
```

## 2. 环境

| 项 | 值 |
|---|---|
| 工作目录 | `D:/nagisa-rs` |
| Conda 环境 | `asr`（已有 nagisa） |
| Python 路径 | `C:/Users/ADMIN/miniconda3/envs/asr/` |
| nagisa 包目录 | `C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages/nagisa/` |
| 模型数据 | 包内 `data/nagisa_v001.hp`（gzip pickle 超参）/ `nagisa_v001.model`（DyNet 文本格式，22 params + 6 lookups）/ `nagisa_v001.dict`（pickle，含 uni2id/bi2id/word2id/id2word/pos2id） |
| 导出产物 | `D:/nagisa-rs/models/weights.safetensors` + `hp.json` + `uni2id.json`/`bi2id.json`/`word2id.json`/`id2word.json`/`pos2id.json` + `word2postags.json`（POS 路径用） |
| cargo / Rust | `cargo 1.96.0`；`src/` 已实现（lib + features/lstm/pos/preprocess/weights 子模块 + tests），`Cargo.toml` 已就绪 |

运行 dynet 测试脚本（模型加载约 1 秒）：

```bash
cd /d D:/nagisa-rs
C:/Users/ADMIN/miniconda3/envs/asr/python scripts/<脚本名>.py
```

## 3. 已确认的事实（请不要再验证）

### 3.1 超参（模型结构）

```text
DIM_UNI=32 DIM_BI=16 DIM_WORD=16 DIM_CTYPE=8 DIM_TAGEMB=16
DIM_HIDDEN=100  WINDOW_SIZE=3  LAYERS=1
VOCAB_SIZE_UNI=3090 VOCAB_SIZE_BI=82114 VOCAB_SIZE_WORD=59260
```

- WS 输入维度 = `(32+16+8)*3 + 16*2 = 200`
- `BiRNN/LSTMBuilder(1, 200, H=100)` → 每个方向内部 `H=50`，全局拼接后 100
- 输出投影 `w_ws (6,100)` / 偏置 `b_ws (6,)` / CRF 转移 `trans (6,6)`
- 双向拼接顺序：`[fwd_h; bwd_h]`（50+50）
- 标签：`0=B 1=I 2=E 3=S 4=sp_s 5=sp_e`
- Viterbi 初始化：`T[sp_s=4]=0`；转移得分用 `T + trans[nxt]`；终止 `T+trans[sp_e=5]`

注：`DIM_HIDDEN=100` 指的应是跨方向拼接后的可见隐藏维度，**黑盒内部每个方向 LSTM 状态维度是 50**（`4*50=200` gate 槽），详见 3.3。

### 3.2 导出映射（`safetensors`）

`safetensors` 里每层的导出键（名称沿用 nagisa 源码参数路径）：

- `uni_emb / bi_emb / ctype_emb / word_emb`：查找表
- `ws_fwd_lstm.W (200,200)` / `ws_fwd_lstm.U (200,50)` / `ws_fwd_lstm.b (200,)`
- `ws_bwd_lstm.W/U/b`：同上形状
- `w_ws (6,100)` / `b_ws (6,)` / `trans (6,6)`

⚠️ **危险**（之前导出脚本踩过的坑，已经修正）：导出时参数路径 `words` 与业务变量 `words` 同名碰撞；`feats()` 的 `append` 放在 `if not sw` 分支里导致只追加空行；`cwin` 窗口计算用 `win-h` 而非 `win`（少一个窗口元素）。当前 `models/` 目录下的 safetensors 已无此类错误，numpy 双精度前向已被 dynet 在单步上验证为 bit-exact。

### 3.3 LSTM 前向（向量化）

- gate 顺序为 4 个 quarter，顺序 `(i,f,o,c)`（输入/遗忘/输出/candidate），长度均为 `H=50`
- **输出 gate 判定**：对 quarter 2 与 dynet 恢复出的 `o_t` 在全部 7 步上求相关，得相关系数 1.0、最大绝对差 `< 1e-2`，确认 quarter 2 = 输出 gate 的预激活值。`o_t = sigmoid(q2)`。
- gate 预激活整体 `z_t = W x_t + U h_{t-1} + b`，shape `(200,)`。

### 3.4 特征提取（预处理）

1. `rstrip` → NFKC → `replace('İ','I')` → `replace(' ', '\u{3000}')`
2. 对长度为 N 的文本，构造 5 个序列（长度 N）：
   - uni ids（bigram/unknown 用 oov）
   - bi ids（`text[i]+text[i+1]`，末尾追加 `<E>`）
   - ctype ids（平假 0 / 片假 1 / 汉 2 / 字母 3 / 数字 4 / 其它 5）
   - `words_starting_at_i`：`j in i..min(i+8,n)` 升序，最长匹配
   - `words_ending_at_i`：`j in i..max(0,i-8)` 降序（最长优先）
3. 对 uni/bi/ctype 各自作 `cwin(l, win=3, pad)`（`pad=1` for uni/bi，`pad=6` for ctype）；`word2id['oov']=17`，`word2id['pad']=1`（**注意不是 0/1 默认假设**）
4. 每个位置 i 的输入 = `concat(uni_emb[uid_window].ravel(), bi_emb[bid_window].ravel(), ctype_emb[cid_window].ravel(), word_emb[start_ids].sum(0).ravel(), word_emb[end_ids].sum(0).ravel())`，得到 `(N,200)` 矩阵

## 4. 当前阻塞（已解决）

### 4.1 现象与根因

**多步发散的根因：dynet VanillaLSTMBuilder 在 f-gate 预激活上加了常数 +1（`forget_gate_bias = 1.0`，见 clab/dynet v2.1 `dynet/nodes-lstm.h:11` 与 `nodes-lstm.cc:154`）。**

源码（`nodes-lstm.cc:104-107`）注释：

```text
gates_i = sigmoid (Wx_i * x_t + Wh_i * h_tm1 + b_i)
gates_f = sigmoid (Wx_f * x_t + Wh_f * h_tm1 + b_f + 1)   ← 缺失项
gates_o = sigmoid (Wx_o * x_t + Wh_o * h_tm1 + b_o)
gates_g =   tanh  (Wx_g * x_t + Wh_g * h_tm1 + b_g)
```

`VanillaLSTMC::forward_dev_impl`（`nodes-lstm.cc:565`）解耦：`c_t = i_t * g_t + f_t * c_tm1`。`VanillaLSTMH::forward_dev_impl`（`nodes-lstm.cc:656`）：`h_t = o_t * tanh(c_t)`。**没有 peephole、没有 layer-norm、没有 forget bias 参数（硬编码 1.0）**。

### 4.2 验证

| 测试 | 结果 |
|---|---|
| 随机 xs 12 步开放循环（`scripts/lstm_locked.py`） | max c_err = 2.86e-6，max h_err = 1.24e-6（float64 噪声） |
| 真实文本 "クラブ" 3 步 fwd+bwd（`scripts/transduce_test.py`，`fg=sig(z[H:2H]+1)`） | 每字符 maxabs ≤ 2.4e-6（float32 噪声） |
| `role_brute.py` 最佳解 `(qf,qi,qc)=(1,0,3)` 与本结论一致 | f-gate 在 q1，+1 后完美；其余排列已无关 |

修复点：所有 LSTM forward 在 f 段预激活后 +1（h 文件 `:24` 声明 `const real forget_gate_bias` 不可改，永远是 1.0）。

### 4.3 为什么单步仍匹配

c_prev = c0 = 0 时，i_t 与 g_t 主导变化，f_t 项（含 +1 偏置后接近 1）影响小；多步后 f_t 项通过 c_{t-1} 持续叠加而显著偏离。

## 5. 已采用路径：直接读 clab/dynet v2.1 C++ 源码

`dynet38-2.2` 是预编译 wheel；通过 README §4.1 列出的对比矩阵定位到 `forget_gate_bias` 常数嫌疑后，直接拉上游：

```bash
curl -sSL https://raw.githubusercontent.com/clab/dynet/2.1/dynet/nodes-lstm.cc
curl -sSL https://raw.githubusercontent.com/clab/dynet/2.1/dynet/nodes-lstm.h
```

关键行（v2.1）：

- `nodes-lstm.h:11` — `forget_gate_bias(1.0)` 构造默认值
- `nodes-lstm.cc:104-107` — 注释明确四个 gate 的激活形式（f 段加 +1）
- `nodes-lstm.cc:154` — `tbvec(fx).slice(indices_f, sizes_1).device(...) += ...constant(forget_gate_bias);`
- `nodes-lstm.cc:565` — `c_t = i_t * g_t + f_t * c_tm1`（解耦）
- `nodes-lstm.cc:656` — `h_t = o_t * tanh(c_t)`

注意上游 tag 是 `2.1`（不是 `v2.1`）。

## 6. 错误教训清单（别再踩）

- `feats()` 内 `if not sw: ...` 分支只追加空行 → `feats` 实际是空的；`append` 必须在 `if` 外。`diag2.py` 早期版本因此跑空。
- `cwin` 误用 `win-h` 作为步长会丢窗口元素，必须用 `win`。
- 导出模型时，param 名称字符串与业务词碰撞（例：`words` 既作参数路径前缀，又作 Python 变量名）。导出脚本中绝不用 `words` 之类的常见变量名。
- 单步验证成功**不代表**细胞更新公式正确。最先做「多步开放循环 + 真值 `h,c` 注入」对比。
- numpy 默认 float64 会掩盖 float32 累积误差；但经过 `pure_f32.py` 验证，**本任务的发散与精度无关**，不必在此花时间。
- 用 `lsp` / `code_actions` 改庞大脚本时容易破坏缩进；小改容易出错就整个文件重写（`_write`）。

## 7. scripts/ 索引

调试期（LSTM 公式锁定）的探索性脚本（`cell_brute.py`、`diag*.py`、`pin*.py`、`recover*.py`、`role_brute.py`、`stepwise*.py` 等 ~31 个）已清理；保留以下两类。

| 脚本 | 说明 |
|---|---|
| `baseline_verify.py` | numpy 全流程（feature→BiLSTM→Viterbi→分词）对照 dynet/nagisa，是 Rust 实现的公式参照 |
| `lstm_locked.py` | 锁定后的多步 LSTM 公式验证（随机 12 步 bit-exact） |
| `transduce_test.py` | 真实文本 "クラブ" 多步 fwd+bwd bit-exact |
| `single_step.py` | 单步真实字符 bit-exact |
| `pos_reference.py` | 导出 POS 路径节点真值（`models/pos_reference.json`）供节点级对照 |
| `extract_srt.py` | 从 SRT 字幕提取文本行（→ `models/srt_lines.json`） |
| `gen_srt_wakati.py` / `gen_srt_tagging.py` | 用 Python nagisa 跑全语料，生成 wakati/tagging 对照基准 |

运行 SRT 语料对照需 conda `asr` 环境（含 `nagisa`），脚本内路径按需调整。

## 8. Rust 端（已完成）

`cargo test` 全绿（16/16）；`wakati` + `tagging` 均与 Python `nagisa` 逐字一致。
节点级抽样：char-BiLSTM `vec_char` maxabs=3.7e-7、pos-BiLSTM `hidden` maxabs=6.5e-7（float32 噪声）。

模块结构：

| 文件 | 职责 |
|---|---|
| `src/lib.rs` | `Tagger` 公共 API（`new` / `wakati` / `tagging` / `postag_words`）+ 4 条 wakati + 4 条 tagging 参考测试 |
| `src/weights.rs` | `hp.json` / 词汇表（含 `word2postags`）/ `weights.safetensors` 加载，强类型封装 |
| `src/preprocess.rs` | `rstrip → NFKC → İ→I → 空格→全角`（`normalize_nfkc` 暴露给 POS 路径的 `preprocess_without_rstrip`） |
| `src/features.rs` | `feature_extraction`：uni/bi/ctype/words 的 cwin 与 start/end 命中 |
| `src/lstm.rs` | LSTM 前向（含 `forget_gate_bias=1.0`）、BiLSTM 拼接、投影、Viterbi、BMES 分词 |
| `src/pos.rs` | POS 路径：char-BiLSTM（`vec_char`）、pos-BiLSTM、`softmax` argmax（`encode_pt`） |

实现要点（与 README §3/§4 的锁定公式严格一致）：

1. ✅ 完成第 5 节路径 B，拿到 dynet 多步 LSTM 真实公式。
2. ✅ numpy 翻译为标准运算（`scripts/lstm_locked.py` 12 步 bit-exact，`scripts/transduce_test.py` 真实 3 步 bit-exact）。
3. ✅ `cargo init --lib`，引入依赖（见下）。
4. ✅ Rust LSTM 前向 + BiLSTM + 投影 + Viterbi + BMES，4/4 匹配。

### 8.1 推荐依赖

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"          # 加载 hp.json + 词汇表
ndarray = "0.15"          # 矩阵运算
safetensors = "0.4"       # 加载 weights.safetensors
anyhow = "1"
unicode-normalization = "0.1"   # NFKC
```

注：不需要 torch/candle 后端——模型很小（~10 MB），纯 ndarray 足够。

### 8.2 Rust 数据中心 FEATS 构造注意事项

- NFKC 用 `unicode_normalization::UnicodeNormalization::nfkc()`
- NFKC 后记得 `replace('İ', 'I')`（U+0130 latin I-with-dot，土耳其语大写）再 `replace(' ', '\u{3000}')`（全角空格）
- `rstrip` 内部对应 Python `text.rstrip`
- BId、UID、ctype 映射用 `HashMap<String,u32>`（或 perfect-hash map，规模不大）
- 字符 3-gram、2-gram 在查 vocab 失败时回退到 oov；注意 `word2id['oov']=17` 不是 0
- OOV bigram 仍可能出现完全未收录的 bi-id——Python nagisa 实际是给一个 `<oov>` 槽位（`bi2id.get('<oov>', 0)`），导出时我们用 0；Rust 实现中也应沿用同样的默认值
- 起始 B / 终止 E 用 `cwin(pad=1)`，ctype 用 `cwin(pad=6)`
- 状态拼接 `[fwd_h; bwd_h]` 顺序已通过 dynet 多方向验证

### 8.3 Rust 推理端建议结构

```rust
pub struct Tagger { ... }
impl Tagger {
    pub fn new(model_dir: &Path) -> Result<Self>;
    pub fn wakati(&self, text: &str) -> Vec<String> { ... }
    pub fn tagging(&self, text: &str) -> TaggingResult { ... }
}
pub struct TaggingResult { pub words: Vec<String>, pub tags: Vec<String> }
```

**先只实现 `wakati`（切分）**：4 条参考文本都只需此 API。POS 可后加。

### 8.4 验证

```bash
cd D:/nagisa-rs
cargo test          # 输出应与 README 4 条参考完全一致
```

每个参考文本都应写入测试文件作为对照基准。

## 9. 关键常量速查

```text
DIM_UNI=32  DIM_BI=16  DIM_WORD=16  DIM_CTYPE=8
DIM_TAGEMB=16  DIM_HIDDEN=100(外层)/50(方向内)  WINDOW_SIZE=3  LAYERS=1
WS_INPUT_DIM=200  TAG_SIZE=6  H_internal=50
PAD_Word=1  oov_word=17
sp_s=4  sp_e=5
fwd_W (200,200)  fwd_U (200,50)  fwd_b (200,)
w_ws (6,100)  b_ws (6,)  trans (6,6)
```

## 10. 何时能结束（已达成）

- ✅ 4 条参考文本 `wakati` 100% 匹配 dynet/nagisa
- ✅ 4 条参考文本 `tagging`（words + postags）100% 匹配 nagisa
- ✅ 额外 18 条文本（含长句、空串、纯空白、数字、ASCII、引号、`$`/`.`/`,`、片假/汉字混合）逐字匹配
- ✅ POS 路径关键节点（`vec_char`、pos-BiLSTM `hidden`、`pid`）与 dynet 抽样 maxabs ≤ 6.5e-7
- ✅ `cargo test` 全绿（16/16）
- ✅ 未再“调整 gate 顺序”或“尝试激活函数变体”——直接套用 §4 锁定的 dynet 多步公式
