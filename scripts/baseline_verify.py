# Baseline numpy forward + gate-order verification against DyNet.
# This script is the Rust implementation's spec: port its exact math to Rust.
import os, sys, json, numpy as np, re, unicodedata
sys.path.insert(0, r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages")
import nagisa, nagisa_utils
import safetensors.numpy as stnp
import dynet as dy

pkg = r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages/nagisa/data"
w = stnp.load_file(r"D:/nagisa-rs/models/weights.safetensors")
vocabs = nagisa_utils.load_data(os.path.join(pkg, "nagisa_v001.dict"))
uni2id, bi2id, word2id, _, _ = vocabs
Hp = json.load(open(r"D:/nagisa-rs/models/hp.json", encoding="utf-8"))
ws = Hp["WINDOW_SIZE"]

_h = re.compile(u'[\u3040-\u309f]')
_k = re.compile(u'[\u30a1-\u30fa]')
_j = re.compile(u'[\u4e00-\u9fa5]')
_a = re.compile(u'[a-zA-Z]')
_n = re.compile(u'[0-9]')


def chartype(ch):
    if _h.search(ch): return 0
    if _k.search(ch): return 1
    if _j.search(ch): return 2
    u = unicodedata.normalize('NFKC', ch)
    if _a.search(u): return 3
    if _n.search(u): return 4
    return 5


OOV_uni = uni2id.get("oov", 0)
OOV_bi = bi2id.get("oov", 0)
oov_id = word2id["oov"]
pad_id = word2id["pad"]


def cwin(l, win, pad):
    h = win // 2
    padded = h * [pad] + l + h * [pad]
    return [padded[i:i + win] for i in range(len(l))]


def conv(ws, idmap, oov):
    return [idmap[word] if word in idmap else oov for word in ws]


def feats_ok(txt):
    lo = txt
    n = len(lo)
    uids = conv(list(lo), uni2id, OOV_uni)
    bids = conv([lo[i] + (u'<E>' if i == n - 1 else lo[i + 1]) for i in range(n)], bi2id, OOV_bi)
    cids = [chartype(c) for c in lo]
    out_s, out_e = [], []
    for i in range(n):
        sw = []
        for j in range(i, min(i + 8, n)):
            sub = lo[i:j + 1]
            if sub in word2id: sw.append(word2id[sub])
        if not sw: sw.append(oov_id)
        out_s.append(sw)
    for i in range(n):
        ew = []
        for j in range(i, max(-1, i - 8), -1):
            sub = lo[j:i + 1]
            if sub in word2id: ew.append(word2id[sub])
        if not ew: ew.append(oov_id)
        out_e.append(ew)
    return [cwin(uids, ws, pad_id), cwin(bids, ws, pad_id), cwin(cids, ws, 6),
            out_s, out_e]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def lstm(xs, W, U, b, H):
    h = np.zeros(H, dtype=xs.dtype)
    c = np.zeros(H, dtype=xs.dtype)
    O = []
    for x in xs:
        z = W @ x + U @ h + b            # (4H,)
        i = sigmoid(z[0:H])
        f = sigmoid(z[H:2*H])
        o = sigmoid(z[2*H:3*H])
        cd = np.tanh(z[3*H:4*H])
        c = f * c + i * cd
        h = o * np.tanh(c)
        O.append(h)
    return np.stack(O)


def build_xs(fe):
    N = len(fe[0])
    parts = []
    for i in range(N):
        vu = w["uni_emb"][fe[0][i]].ravel()
        vb = w["bi_emb"][fe[1][i]].ravel()
        vc = w["ctype_emb"][fe[2][i]].ravel()
        vs = w["word_emb"][fe[3][i]].sum(0).ravel()
        ve = w["word_emb"][fe[4][i]].sum(0).ravel()
        parts.append(np.concatenate([vu, vb, vc, vs, ve]))
    return np.stack(parts)


def encode_np(xs):
    F, Uf, bf = w["ws_fwd_lstm.W"], w["ws_fwd_lstm.U"], w["ws_fwd_lstm.b"]
    B, Ub, bb = w["ws_bwd_lstm.W"], w["ws_bwd_lstm.U"], w["ws_bwd_lstm.b"]
    o_f = lstm(xs, F, Uf, bf, 50)
    o_b = lstm(xs[::-1], B, Ub, bb, 50)[::-1]
    big = np.concatenate([o_f, o_b], 1)   # L x 100
    return big @ w["w_ws"].T + w["b_ws"]


def viterbi(obs, trans):
    TT = np.full(6, -1e10, dtype=obs.dtype); TT[4] = 0.0
    bp = []
    for o in obs:
        cur = []; got = []
        for nxt in range(6):
            v = TT + trans[nxt]
            bj = int(np.argmax(v)); got.append(v[bj]); cur.append(bj)
        TT = np.array(got) + o
        bp.append(cur)
    term = TT + trans[5]; best = int(np.argmax(term)); path = [best]
    for b in reversed(bp): best = b[best]; path.append(best)
    return np.array(path[:-1][::-1])


def seg(chars, tags):
    words = []; p = ""
    for ch, t in zip(chars, tags):
        if t == 3: words.append(ch)
        elif t == 2: p += ch; words.append(p); p = ""
        else: p += ch
    return words


dy.renew_cg()
t = nagisa.Tagger()._model
refs = [
    ("シンデレライライゼロ", ["シンデレ", "ライライゼロ"]),
    ("クラブリベリオンシンデレライライゼロ", ["クラブリベリオン", "シンデレ", "ライライゼロ"]),
    ("女子アナの仕事に耐える。辛抱大工です。", ["女子", "アナ", "の", "仕事", "に", "耐える", "。", "辛抱", "大工", "です", "。"]),
    ("本日は島根県にある有名な人気ラーメン店にやってきました。",
     ["本日", "は", "島根", "県", "に", "ある", "有名", "な", "人気", "ラーメン", "店", "に", "やっ", "て", "き", "まし", "た", "。"]),
]
for txt, exp in refs:
    f2 = nagisa_utils.feature_extraction(txt, uni2id, bi2id, word2id, ws)
    fe = feats_ok(txt)
    fe_ok_bits = 0
    for ai, (a, b) in enumerate(zip(fe, f2)):
        if list(map(list, a)) == list(map(list, b)):
            fe_ok_bits |= (1 << ai)
    xs = build_xs(fe)
    obs_np = encode_np(xs)
    tags_np = viterbi(obs_np, w["trans"].astype(obs_np.dtype))
    got_np = seg(list(txt), tags_np)
    dy.renew_cg()
    obs_dy_exprs = t.encode_ws(f2)
    obs_dy = np.stack([ob.npvalue() for ob in obs_dy_exprs]).astype(obs_np.dtype)
    diff = obs_np - obs_dy
    print("%-28s bits=%d expmatch=%s pymatch=%s maxdiff=%.3g meandiff=%.3g le6=%d le4=%d/%d" %
          ("'" + txt[:12],
           fe_ok_bits, got_np == exp, got_np == nagisa.wakati(txt),
           float(np.max(np.abs(diff))), float(np.mean(np.abs(diff))),
           int(np.sum(np.abs(diff) <= 1e-6)), int(np.sum(np.abs(diff) <= 1e-4)), diff.size))
    print("    np:", got_np)
    print("    py:", nagisa.wakati(txt))
