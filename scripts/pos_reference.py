# Dump intermediate POS-tagging values from real nagisa for node-by-node Rust validation.
# Outputs JSON with: tids (after noun heuristic), vec_char (char-BiLSTM final), vec_word, vec_tag,
# pos_bilstm hidden, softmax prob, predicted pid.
import os, sys, json, numpy as np
sys.path.insert(0, r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages")
import nagisa, nagisa_utils
import dynet as dy

pkg = r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages/nagisa/data"
vocabs = nagisa_utils.load_data(os.path.join(pkg, "nagisa_v001.dict"))
uni2id, bi2id, word2id, pos2id, word2postags = vocabs
tagger = nagisa.Tagger()
m = tagger._model
use_noun_heuristic = tagger.use_noun_heuristic
id2pos = {v: k for k, v in pos2id.items()}

def preprocess_words(words):
    out = []
    for w in words:
        if w == " " or w == "\u3000":
            out.append(nagisa_utils.preprocess_without_rstrip(w))
        else:
            out.append(nagisa_utils.preprocess(w))
    return out

def build_X(words):
    words = preprocess_words(words)
    wids = nagisa_utils.conv_tokens_to_ids(words, word2id)
    cids = [nagisa_utils.conv_tokens_to_ids([c for c in w], uni2id) for w in words]
    tids = []
    for w in words:
        w2p = set(word2postags.get(w, [0]))
        if use_noun_heuristic and w.isalnum():
            if 0 in w2p:
                w2p.remove(0)
            w2p.add(pos2id[u"名詞"])
        tids.append(list(w2p))
    return [cids, wids, tids]

def encode_pt_node_dump(X):
    dy.renew_cg()
    length = len(X[0])
    ipts = []
    vec_chars_dump = []
    vec_words_dump = []
    vec_tags_dump = []
    for i in range(length):
        cids = X[0][i]; wid = X[1][i]; tids = X[2][i]
        vec_char_expr = m.char_seq_model.transduce([m.UNI[cid] for cid in cids])[-1]
        vec_chars_dump.append(np.array(vec_char_expr.npvalue(), dtype=np.float32))
        vec_tags = []
        for tid in tids:
            if tid == 0:
                zero = dy.inputVector(np.zeros(m.dim_tag_emb))
                vec_tags.append(zero)
            else:
                vec_tags.append(m.POS[tid])
        vec_tag_expr = dy.esum(vec_tags)
        if wid == 0:
            vec_word_expr = dy.inputVector(np.zeros(m.dim_word))
        else:
            vec_word_expr = m.WORD[wid]
        vec_words_dump.append(np.array(vec_word_expr.npvalue(), dtype=np.float32))
        vec_tags_dump.append(np.array(vec_tag_expr.npvalue(), dtype=np.float32))
        vec_at_i = dy.concatenate([vec_word_expr, vec_char_expr, vec_tag_expr])
        ipts.append(vec_at_i)
    hiddens = m.pos_model.transduce(ipts)
    probs = [dy.softmax(m.w_pos * h + m.b_pos) for h in hiddens]
    hiddens_dump = np.stack([np.array(h.npvalue(), dtype=np.float32) for h in hiddens])
    probs_dump = np.stack([np.array(p.npvalue(), dtype=np.float32) for p in probs])
    pids = [int(np.argmax(p)) for p in probs_dump]
    return {
        "tids": X[2],
        "wids": X[1],
        "cids": X[0],
        "vec_char": vec_chars_dump,
        "vec_word": vec_words_dump,
        "vec_tag": vec_tags_dump,
        "hidden": hiddens_dump,
        "prob": probs_dump,
        "pid": pids,
    }

refs = [
    "本日は島根県にある有名な人気ラーメン店にやってきました。",
    "AIの進化が凄まじい。",
    "シンデレライライゼロ",
    "田中さんは毎朝7時にコーヒーを飲みながらニュースを見ます。",
]
out = {}
for txt in refs:
    words = tagger.wakati(txt)
    X = build_X(words)
    dump = encode_pt_node_dump(X)
    postags = [id2pos[p] for p in dump["pid"]]
    out[txt] = {
        "words": words,
        "tids": dump["tids"],
        "wids": dump["wids"],
        "cids": dump["cids"],
        "vec_char": np.stack(dump["vec_char"]).tolist(),
        "vec_word": np.stack(dump["vec_word"]).tolist(),
        "vec_tag": np.stack(dump["vec_tag"]).tolist(),
        "hidden": dump["hidden"].tolist(),
        "prob": dump["prob"].tolist(),
        "pid": dump["pid"],
        "postags": postags,
    }
with open(r"D:/nagisa-rs/models/pos_reference.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("wrote pos_reference.json")
for txt, d in out.items():
    print(f"--- {txt}")
    print("  words:", d["words"])
    print("  postags:", d["postags"])
    print("  tids[0:3]:", d["tids"][:3])
    print("  cids[0:2]:", d["cids"][:2])
