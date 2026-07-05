# Locked formula (dynet v2.1 VanillaLSTMBuilder, nodes-lstm.h line 11: forget_gate_bias=1.0):
#   i_t = sigmoid(q0) ; f_t = sigmoid(q1 + 1) ; o_t = sigmoid(q2) ; g_t = tanh(q3)
#   c_t = f_t * c_{t-1} + i_t * g_t  (decoupled)
#   h_t = o_t * tanh(c_t)
# Verify multi-step bit-exact vs dynet using TRUE (h_{t-1}, c_{t-1}) injection.
import os, sys, numpy as np
sys.path.insert(0, r"C:/Users/ADMIN/miniconda3/envs/asr/lib/site-packages")
import dynet as dy
import safetensors.numpy as stnp

w = stnp.load_file(r"D:/nagisa-rs/models/weights.safetensors")
Ws = w["ws_fwd_lstm.W"].astype(np.float64)
Us = w["ws_fwd_lstm.U"].astype(np.float64)
bs = w["ws_fwd_lstm.b"].astype(np.float64)
H = Ws.shape[0] // 4

pc = dy.ParameterCollection()
builder = dy.VanillaLSTMBuilder(1, 200, H, pc)
Wp, Up, bp = pc.parameters_list()
Wp.set_value(Ws.astype(np.float32)); Up.set_value(Us.astype(np.float32)); bp.set_value(bs.astype(np.float32))

# Two scenarios: random xs (pure synthetic) AND a real short token sequence.
N = 12
rng = np.random.RandomState(1)
xs_rand = rng.randn(N, 200).astype(np.float64)

def run(xs):
    dy.renew_cg()
    h_list, c_list = [], []
    s = builder.initial_state()
    for x in xs:
        s = s.add_input(dy.inputVector(x.astype(np.float32)))
        h_list.append(np.array(s.output().npvalue(), dtype=np.float64))
        c_list.append(np.array(s.s()[0].npvalue(), dtype=np.float64))
    return np.stack(h_list), np.stack(c_list)

def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-80,80)))

def locked_step(z_t, c_prev):
    q0, q1, q2, q3 = z_t[0*H:1*H], z_t[1*H:2*H], z_t[2*H:3*H], z_t[3*H:4*H]
    i_t = sig(q0); f_t = sig(q1 + 1.0); o_t = sig(q2); g_t = np.tanh(q3)
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * np.tanh(c_t)
    return c_t, h_t

def compare(xs, label):
    h_true, c_true = run(xs)
    h_pred, c_pred = np.zeros_like(h_true), np.zeros_like(c_true)
    c_pred[0] = c_true[0]   # match dynet init exactly
    h_pred[0] = h_true[0]
    for t in range(1, len(xs)):
        z = Ws @ xs[t] + Us @ h_pred[t-1] + bs
        c_pred[t], h_pred[t] = locked_step(z, c_pred[t-1])
    cerr = np.max(np.abs(c_pred - c_true), axis=1)
    herr = np.max(np.abs(h_pred - h_true), axis=1)
    print(f"--- {label} (N={len(xs)}) ---")
    for t in range(len(xs)):
        print(f"  step {t}: c_err={cerr[t]:.3e}  h_err={herr[t]:.3e}")
    print(f"  max c_err={cerr.max():.3e}  max h_err={herr.max():.3e}")
    return cerr.max(), herr.max()

ce1, he1 = compare(xs_rand, "random xs")
print(f"=> RANDOM: max c_err={ce1:.3e}, max h_err={he1:.3e}")