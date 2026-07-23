# nagisa-rs

Pure-Rust port of [nagisa](https://github.com/taishi-i/nagisa) — a Japanese word segmentation and part-of-speech tagging library based on a DyNet BiLSTM-CRF model.

This crate reproduces Python `nagisa`'s output **byte-for-byte**. Every key computational node (BiLSTM hidden states, softmax outputs, CRF decoding) has been verified against the original DyNet implementation; on a real 1,228-line subtitle corpus, both `wakati` and `tagging` match Python nagisa **100.0000%** line- and token-wise.

## Features

- **`wakati`** — word segmentation, equivalent to `nagisa.wakati(text)`
- **`tagging`** — word segmentation + POS tagging, equivalent to `nagisa.tagging(text)` (`.words` / `.postags`)
- **`postag_words`** — POS-tag a pre-tokenized word list, equivalent to `nagisa.postagging(words)`
- **No ML runtime dependency** — pure `ndarray` linear algebra, ~10 MB model, fast startup
- **Bit-exact** with the reference Python implementation

## Quick start

```rust
use nagisa_rs::Tagger;

fn main() {
    // Preferred: load the model compiled into the crate (~25 MB binary growth).
    // No external files or download required.
    let tagger = Tagger::embedded().expect("failed to load embedded model");
    // Alternative: load from a directory of weights/dicts.
    // let tagger = Tagger::new("models").expect("failed to load model");

    // Word segmentation
    let words = tagger.wakati("ネコは魚を食べるのが好きです。");
    assert_eq!(words, vec!["ネコ", "は", "魚", "を", "食べる", "の", "が", "好き", "です", "。"]);

    // Word segmentation + POS tagging
    let result = tagger.tagging("私は明日東京タワーに行きます。");
    for (word, pos) in result.words.iter().zip(result.postags.iter()) {
        println!("{word}/{pos}");
    }
    // 私/代名詞  は/助詞  明日/名詞  東京/名詞  タワー/名詞  に/助詞
    // 行き/動詞  ます/助動詞  。/補助記号
}
```

## Installation

The crate is not published to crates.io yet. Use it from a git dependency:

```toml
[dependencies]
nagisa_rs = { git = "https://github.com/eclipse005/nagisa-rs.git" }
```

The model weights and dictionaries are **compiled into the crate** via `Tagger::embedded()`.
The on-disk `models/` tree remains available for `Tagger::new` and for development tests.

## Build & test

Requires Rust 1.85+ (edition 2024).

```bash
cargo build
cargo test        # 16 unit tests + doctest, all green
```

Tests load the bundled `models/` via a path relative to the crate root (`CARGO_MANIFEST_DIR/models`), so they run out of the box after clone.

## How it works

The model is the same architecture as upstream nagisa: a character/word-feature BiLSTM with a CRF decoding layer for word segmentation, plus a separate char-BiLSTM + word/tag-feature BiLSTM with softmax for POS tagging.

```
                ┌────────────  wakati (word segmentation)  ────────────┐
text → preprocess │ → features → WS-BiLSTM → linear → CRF Viterbi → BMES │ → words
                └──────────────────────────────────────────────────────┘

                ┌────────────  tagging (POS tagging)  ──────────────────┐
      words  →  │ → char-BiLSTM + word/tag features → pos-BiLSTM → softmax │ → postags
                └──────────────────────────────────────────────────────┘
```

### Module layout

| Module | Responsibility |
|---|---|
| [`src/lib.rs`](src/lib.rs) | `Tagger` public API (`new` / `wakati` / `tagging` / `postag_words`) + reference tests |
| [`src/weights.rs`](src/weights.rs) | Load `hp.json`, vocab dictionaries, `weights.safetensors` |
| [`src/preprocess.rs`](src/preprocess.rs) | `rstrip → NFKC → İ→I → space→ideographic space` normalization |
| [`src/features.rs`](src/features.rs) | Character/word feature extraction (uni/bi/ctype context windows, dictionary hits) |
| [`src/lstm.rs`](src/lstm.rs) | LSTM forward (with `forget_gate_bias=1.0`), BiLSTM, projection, Viterbi, BMES segmenter |
| [`src/pos.rs`](src/pos.rs) | POS pipeline: char-BiLSTM, pos-BiLSTM, softmax (`encode_pt`) |

### The locked LSTM formula

The single hardest detail in porting was matching DyNet's `VanillaLSTMBuilder` exactly. The cell update uses a hardcoded **forget-gate bias of +1.0** (DyNet `nodes-lstm.h:11`, `forget_gate_bias = 1.0`):

```
z = W·x + U·h_prev + b                          # gate pre-activations (4H)
i = σ(z[0:H])     f = σ(z[H:2H] + 1.0)          # ← the +1 is the locked term
o = σ(z[2H:3H])   g = tanh(z[3H:4H])
c = f · c_prev + i · g
h = o · tanh(c)
```

Without the `+1.0`, single-step outputs still matched but multi-step sequences diverged through the recurrent cell state. See the original [dynet v2.1 `nodes-lstm.cc`](https://github.com/clab/dynet/blob/2.1/dynet/nodes-lstm.cc) for the authoritative source.

## Verification

Accuracy is validated at two levels:

**Node-level** (against the live DyNet model, float32):
- WS BiLSTM: matches `scripts/lstm_locked.py` to ≤ 1.2e-6 over 12 random steps
- POS char-BiLSTM `vec_char`: maxabs = 3.7e-7
- POS BiLSTM `hidden`: maxabs = 6.5e-7

**Corpus-level** (against Python `nagisa`):
- 4 hand-curated reference sentences — exact match
- 18 additional sentences (digits, ASCII, punctuation, mixed scripts, empty input) — exact match
- **Real subtitle corpus: 1,228 lines / 10,914 tokens — `wakati` and `tagging` both 100.0000% line- and token-exact**

The corpus-validation tooling lives in [`scripts/`](scripts/): `extract_srt.py` pulls text from an SRT file, `gen_srt_wakati.py` / `gen_srt_tagging.py` produce the Python baseline, and the Rust side compares. Re-running on a new corpus takes the same three steps.

## Model files

`models/` contains everything needed for inference, exported from the official `nagisa_v001` model:

| File | Purpose |
|---|---|
| `weights.safetensors` | All LSTM/projection/CRF/POS weights (float32) |
| `hp.json` | Hyperparameters (dims, window size, layer count) |
| `uni2id.json` / `bi2id.json` | Character unigram / bigram vocabularies |
| `word2id.json` / `id2word.json` | Word vocabulary |
| `pos2id.json` | POS-tag vocabulary (24 tags) |
| `word2postags.json` | Word → allowed POS-tag sets (for POS-tagging features) |

## Credits

- Original model and architecture: [Taishi Ikeda's `nagisa`](https://github.com/taishi-i/nagisa)
- Trained weights are the upstream `nagisa_v001` release, re-exported to safetensors for this Rust port

## License

This Rust port follows the upstream project's licensing. The bundled model weights (`models/`) originate from the nagisa project; please respect its terms when redistributing.
