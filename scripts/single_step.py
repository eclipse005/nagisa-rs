# Single-step comparison on real char "ク": fresh fwd/bwd VanillaLSTMBuilder vs dy encode_ws.
import os, sys, numpy as np, re, unicodedata
sys.path.insert(0, r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages")
import dynet as dy
import nagisa_utils, nagisa
import safetensors.numpy as stnp
import json

pkg = r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages/nagisa/data"
w = stnp.load_file(r"D:/nagisa-rs/models/weights.safetensors")
vocabs = nagisa_utils.load_data(os.path.join(pkg, "nagisa_v001.dict"))
uni2id, bi2id, word2id, _, _ = vocabs
Hp = json.load(open(r"D:/nagisa-rs/models/hp.json", encoding="utf-8")); wshp = Hp["WINDOW_SIZE"]

_h = re.compile(u'[\u3040-\u309f]'); _k = re.compile(u'[\u30a1-\u30fa]'); _jj = re.compile(u'[\u4e00-\u9fa5]')
_a = re.compile(u'[a-zA-Z]'); _n = re.compile(u'[0-9]')
def chartype(ch):
    if _h.search(ch): return 0
    if _k.search(ch): return 1
    if _jj.search(ch): return 2
    u = unicodedata.normalize('NFKC', ch)
    if _a.search(u): return 3
    if _n.search(u): return 4
    return 5
OOV_uni = uni2id.get("oov", 0); OOV_bi = bi2id.get("oov", 0); oov_id = word2id["oov"]; pad_id = word2id["pad"]

def cwin(l, win, pad):
    hh = win // 2; padded = hh*[pad]+l+hh*[pad]; return [padded[i:i+win] for i in range(len(l))]
def conv(ws, idmap, oov): return [idmap[word] if word in idmap else oov for word in ws]

def build_xs_for(txt):
    lo = txt; n = len(lo)
    uids = conv(list(lo), uni2id, OOV_uni)
    bids = conv([lo[i]+(u'<E>' if i==n-1 else lo[i+1]) for i in range(n)], bi2id, OOV_bi)
    cids = [chartype(c) for c in lo]
    ss, ee = [], []
    for i in range(n):
        sw = [word2id[lo[i:j+1]] for j in range(i, min(i+8, n)) if lo[i:j+1] in word2id]
        if not sw: sw.append(oov_id)
        ss.append(sw)
    for i in range(n):
        ew = [word2id[lo[j:i+1]] for j in range(i, max(-1, i-8), -1) if lo[j:i+1] in word2id]
        if not ew: ew.append(oov_id)
        ee.append(ew)
    fe = [cwin(uids, wshp, pad_id), cwin(bids, wshp, pad_id), cwin(cids, wshp, 6), ss, ee]
    N = len(fe[0]); out = []
    for i in range(N):
        vu=w["uni_emb"][fe[0][i]].ravel(); vb=w["bi_emb"][fe[1][i]].ravel(); vc=w["ctype_emb"][fe[2][i]].ravel()
        vs=w["word_emb"][fe[3][i]].sum(0).ravel(); ve=w["word_emb"][fe[4][i]].sum(0).ravel()
        out.append(np.concatenate([vu,vb,vc,vs,ve]))
    return np.stack(out)

F = w["ws_fwd_lstm.W"].astype(np.float32); Uf=w["ws_fwd_lstm.U"].astype(np.float32); bf=w["ws_fwd_lstm.b"].astype(np.float32)
Bw=w["ws_bwd_lstm.W"].astype(np.float32); Ub_=w["ws_bwd_lstm.U"].astype(np.float32); bbw=w["ws_bwd_lstm.b"].astype(np.float32)
H = F.shape[0]//4

def fresh_step(W_, U_, b_, x0):
    pc = dy.ParameterCollection(); bd = dy.VanillaLSTMBuilder(1, 200, H, pc)
    Wp, Up, bp = pc.parameters_list(); Wp.set_value(W_); Up.set_value(U_); bp.set_value(b_)
    dy.renew_cg()
    o = bd.initial_state().add_input(dy.inputVector(x0))
    return np.array(o.output().npvalue(), dtype=np.float64), np.array(o.s()[0].npvalue(), dtype=np.float64)

for txt in ["ク", "女", "A", "1"]:
    xs = build_xs_for(txt)
    print("===", repr(txt), "xs shape", xs.shape)
    hA, cA = fresh_step(F, Uf, bf, xs[0])
    hB, cB = fresh_step(Bw, Ub_, bbw, xs[0])
    big = np.concatenate([hA[None], hB[None]], 1)
    obs_fresh = (big @ w["w_ws"].T.astype(np.float32) + w["b_ws"].astype(np.float32)).astype(np.float64)
    dy.renew_cg()
    ref = nagisa.Tagger()._model
    f2 = nagisa_utils.feature_extraction(txt, uni2id, bi2id, word2id, wshp)
    obs_dy = np.stack([ob.npvalue() for ob in ref.encode_ws(f2)]).astype(np.float64)
    print("  fresh obs[0]:", obs_fresh[0].round(3))
    print("  dy    obs[0]:", obs_dy[0].round(3))
    print("  maxabs diff:", float(np.max(np.abs(obs_fresh[0]-obs_dy[0]))))
