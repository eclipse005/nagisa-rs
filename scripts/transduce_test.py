# Test whether fresh BiRNNBuilder.transduce matches dy when given ALL inputs at once (the real path).
import os, sys, numpy as np
sys.path.insert(0, r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages")
import dynet as dy
import safetensors.numpy as stnp
import nagisa_utils, json

pkg = r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages/nagisa/data"
w = stnp.load_file(r"D:/nagisa-rs/models/weights.safetensors")
Hp = json.load(open(r"D:/nagisa-rs/models/hp.json", encoding="utf-8"))
vocabs = nagisa_utils.load_data(os.path.join(pkg, "nagisa_v001.dict"))
uni2id, bi2id, word2id, _, _ = vocabs
wshp = Hp["WINDOW_SIZE"]
_h = __import__("re").compile(u'[\u3040-\u309f]'); _k = __import__("re").compile(u'[\u30a1-\u30fa]')
_jj = __import__("re").compile(u'[\u4e00-\u9fa5]'); _a = __import__("re").compile(u'[a-zA-Z]'); _nn = __import__("re").compile(u'[0-9]')
import unicodedata
def chartype(ch):
    if _h.search(ch): return 0
    if _k.search(ch): return 1
    if _jj.search(ch): return 2
    u = unicodedata.normalize('NFKC', ch)
    if _a.search(u): return 3
    if _nn.search(u): return 4
    return 5
OOV_uni = uni2id.get("oov", 0); OOV_bi = bi2id.get("oov", 0); oov_id = word2id["oov"]; pad_id = word2id["pad"]

def cwin(l, win, pad):
    hh = win//2; padded=hh*[pad]+l+hh*[pad]; return [padded[i:i+win] for i in range(len(l))]
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

F=w["ws_fwd_lstm.W"].astype(np.float32); Uf=w["ws_fwd_lstm.U"].astype(np.float32); bf=w["ws_fwd_lstm.b"].astype(np.float32)
Bw=w["ws_bwd_lstm.W"].astype(np.float32); Ub_=w["ws_bwd_lstm.U"].astype(np.float32); bbw=w["ws_bwd_lstm.b"].astype(np.float32)
H=F.shape[0]//4

# Build a fresh BiRNNBuilder with same param order as real model via dy.LSTMBuilder factory, copy params
import nagisa.model as mmod
real_model_obj = mmod.Model(Hp, pkg + "/nagisa_v001.model")
rp = real_model_obj.model

# Use transduce on the real model
dy.renew_cg()
txt = "クラブ"  # short
f2 = nagisa_utils.feature_extraction(txt, uni2id, bi2id, word2id, wshp)
obs_bs = np.stack([ob.npvalue() for ob in real_model_obj.encode_ws(f2)]).astype(np.float32)

# My numpy full encode (loop lstm, concat fwd/bwd)
xs = build_xs_for(txt)
def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-80,80)))
def lstm_(xs_, W, U, b):
    hh=np.zeros(H); c=np.zeros(H); O=[]
    for x in xs_:
        z=W@x+U@hh+b
        ig=sig(z[0:H]); fg=sig(z[H:2*H]+1.0); og=sig(z[2*H:3*H]); cg=np.tanh(z[3*H:4*H])  # +1 forget bias
        c=fg*c+ig*cg; hh=og*np.tanh(c); O.append(hh)
    return np.stack(O)
o_f=lstm_(xs.astype(np.float32),F,Uf,bf)
o_b=lstm_(xs[::-1].astype(np.float32),Bw,Ub_,bbw)[::-1]
big=np.concatenate([o_f,o_b],1)
obs_np=(big@w["w_ws"].T.astype(np.float32)+w["b_ws"].astype(np.float32)).astype(np.float32)
print("txt", repr(txt))
for t in range(len(txt)):
    print(f"  ch{dy if False else ''} {obs_bs[t].round(3)}")
    print(f"  np {obs_np[t].round(3)}")
    print(f"  maxabs ch{t}:", float(np.max(np.abs(obs_bs[t]-obs_np[t]))))
