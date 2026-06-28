"""V21 快速考试评测 — 256D架构, 加载V21_best.pt"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os, time
sys.path.insert(0,'C:/ai')
from utils.config import *
from P7_cross_sent.model import P7WordRouter2048
from P3_word_attr.stack import P3AttributeStack

_p3s = P3AttributeStack()

# ── 模型定义 (256D全量) ──
class P6_Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_words = 128
        self.encoder = nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(128,256)*2.0)
        self.heads = nn.ModuleList([nn.Linear(256,256) for _ in range(128)])
    def forward(self, sv, cw, gate=None):
        h = self.encoder(sv)
        if gate is not None: h = h * gate
        w = [self.heads[i](h+self.pos_embed[i].unsqueeze(0)) for i in range(self.max_words)]
        return torch.stack(w,1).matmul(cw.T)

class ABC_StageA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(384,256),nn.GELU(),nn.Linear(256,128),nn.GELU())
        self.pool = nn.Linear(128,128); self.classifier = nn.Linear(128,15)
    def forward(self, attr_vec, word_out):
        x = torch.cat([attr_vec, word_out], dim=-1)
        h = self.encoder(x).mean(dim=1)
        return self.classifier(self.pool(h)), self.pool(h)

class ABC_StageB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fuse = nn.Sequential(nn.Linear(128+128+128,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,128))
    def forward(self, a_hidden, attr_vec, p3l_feat):
        return self.fuse(torch.cat([a_hidden,attr_vec.mean(dim=1),p3l_feat],-1))

class ABC_StageC(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine = nn.Sequential(nn.Linear(128+5,96),nn.GELU(),nn.Linear(96,96),nn.GELU(),nn.Linear(96,48))
    def forward(self, content, attr_vec):
        return self.refine(torch.cat([content, attr_vec[:,:,34:39].mean(dim=1)],-1))

class TE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
    def forward(self,x): return torch.tanh(self.net(x))

class TM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(256)*0.1)
    def forward(self,x): return torch.sigmoid(self.bias+x)

def words_to_attr_vec(w, max_n=80):
    p = _p3s.process_sentence(w[:max_n])
    dcts = [pkt.to_dict() for pkt in p]
    vec = torch.zeros(max_n, 128)
    for i, d in enumerate(dcts[:max_n]):
        bt = d.get('basic_type',('',0.0))
        if isinstance(bt,tuple) and bt[0]:
            m = {"noun":0,"verb":1,"adj":2,"adv":3,"pronoun":4,"quantifier":5,
                 "preposition":6,"conjunction":7,"auxiliary":8,"interjection":9}
            vec[i,m.get(bt[0],9)]=bt[1] if len(bt)>=2 else 0.8
        sem=d.get('semantic_types',[])
        if isinstance(sem,list):
            for s in sem[:2]:
                if isinstance(s,(tuple,list)) and len(s)>=2:
                    sl={'人物':0,'地点':1,'时间':2,'物体':3,'行为':4,'状态':5,
                        '数量':6,'程度':7,'方位':8,'方式':9,'原因':10,'结果':11,
                        '目的':12,'条件':13}.get(s[0],-1)
                    if sl>=0:vec[i,10+sl]=s[1]
        syn=d.get('syntax_candidates',[])
        if isinstance(syn,list) and syn:
            s0=syn[0]
            if isinstance(s0,(tuple,list)) and len(s0)>=2:
                sl={'主语':0,'谓语':1,'宾语':2,'定语':3,'状语':4,
                    '补语':5,'兼语':6,'连动':7,'同位':8,'独立':9}.get(s0[0],-1)
                if sl>=0:vec[i,24+sl]=s0[1]
        pol=d.get('polarity',('neutral',0.0,'none'))
        if isinstance(pol,tuple) and len(pol)>=1:
            vec[i,34]=1.0 if pol[0]=='positive' else (-1.0 if pol[0]=='negative' else 0.0)
            vec[i,35]=pol[1] if len(pol)>=2 else 0.0
        conn=d.get('conn_type','')
        cslots={'cause':0,'adversative':1,'coordinate':2,'conditional':3,
                'progressive':4,'concessive':5,'alternative':6,'sequential':7,
                'summary':8,'example':9,'purpose':10}
        if conn in cslots:vec[i,42+cslots[conn]]=d.get('conn_confidence',0.85)
        mod=d.get('mod_type','')
        mslots={'adjective':0,'adverb_manner':1,'scope':2,'attributive':3,'adverbial':4,'complement':5}
        if mod in mslots:vec[i,56+mslots[mod]]=d.get('mod_confidence',0.8)
        if d.get('is_comparative',False):vec[i,63]=0.8
        vec[i,31]=i/max(max_n,1)
    return vec

# ── 加载 ──
ckpt_path = 'C:/ai/P1_char_word/checkpoints/V21_best.pt'
ckpt = torch.load(ckpt_path, map_location=DEVICE)
c2i = ckpt['c2i']; i2c = {i:c for c,i in c2i.items()}
V = len(c2i); epoch = ckpt.get('epoch','?'); best_loss = ckpt.get('loss','?')
print(f"═══ V21 考试评测 ═══")
print(f"  E{epoch} loss={best_loss:.4f} V={V}")

ce = nn.Embedding(V, 256, padding_idx=0).to(DEVICE); ce.load_state_dict(ckpt['char_embed'], strict=False)
p7 = P7WordRouter2048(heads=16, head_dim=16, word_dim=256, inner_dim=256, max_len=128, num_groups=4).to(DEVICE)
p6 = P6_Tied().to(DEVICE)
p7_in = nn.Linear(384, 256, bias=False).to(DEVICE)
abcA = ABC_StageA().to(DEVICE); abcB = ABC_StageB().to(DEVICE); abcC = ABC_StageC().to(DEVICE)
explore = TE().to(DEVICE); meta = TM().to(DEVICE)
abc_to_gate = nn.Linear(48, 12, bias=False).to(DEVICE)
sent_to_gate = nn.Linear(256, 12, bias=False).to(DEVICE)

p7.load_state_dict(ckpt['p7'], strict=False)
p6.load_state_dict(ckpt['p6'], strict=False)
p7_in.load_state_dict(ckpt['p7in'], strict=False)
abcA.load_state_dict(ckpt.get('abcA', {}), strict=False)
abcB.load_state_dict(ckpt.get('abcB', {}), strict=False)
abcC.load_state_dict(ckpt.get('abcC', {}), strict=False)
explore.load_state_dict(ckpt.get('explore', {}), strict=False)
meta.load_state_dict(ckpt.get('meta', {}), strict=False)
abc_to_gate.load_state_dict(ckpt.get('abc_to_gate', {}), strict=False)
sent_to_gate.load_state_dict(ckpt.get('sent_to_gate', {}), strict=False)
for m in [ce,p7,p6,p7_in,abcA,abcB,abcC,explore,meta,abc_to_gate,sent_to_gate]: m.eval()

# 考试集 (从V21训练的分割)
exam = torch.load('C:/ai/P1_char_word/checkpoints/V19_exam_set.pt', map_location='cpu')
# 但V21用了不同c2i, 用训练集首对测试
print(f"\n═══ 训练集首对测试 ═══")
import re, random
SENT_SPLIT = re.compile(r'[。！？；\n]')
all_pairs = []
with open("C:/ai/data/public/public_combined.txt", 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        a, b = line.split('\t', 1)
        sa = [p.strip() for p in SENT_SPLIT.split(a) if len(p.strip())>=3]
        sb = [p.strip() for p in SENT_SPLIT.split(b) if len(p.strip())>=3]
        for xa, xb in zip(sa, sb):
            wa = [c for c in xa if c.strip() and ord(c)>32]
            wb = [c for c in xb if c.strip() and ord(c)>32]
            if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                all_pairs.append((wa, wb))
random.seed(42)
sampled = random.sample(all_pairs, 1500)
all_chars_v21 = set()
for A, B in sampled: all_chars_v21.update(A); all_chars_v21.update(B)
encoded = [([c2i.get(c,0) for c in A], [c2i.get(c,0) for c in B], A, B) for A, B in sampled]
random.shuffle(encoded)
n = len(encoded)
train_set = encoded[:int(n*0.80)]; exam_set = encoded[int(n*0.80)+int(n*0.10):]

vocab = ce.weight.detach()
total_ok = 0; total_n = 0
n_test = min(10, len(exam_set))

for idx in range(n_test):
    A_ids, B_ids, A, B = exam_set[idx]
    na = len(A); nb = len(B)
    with torch.no_grad():
        A_emb = ce(torch.tensor([c2i.get(c,0) for c in A], device=DEVICE))
        attr_a = words_to_attr_vec(A).to(DEVICE)[:na, :]
        Av_rich = torch.cat([A_emb, attr_a], dim=-1).unsqueeze(0)
        p7._loss_vec = None; p7.ce_ema = 0.1  # 收敛状态
        wo, sv, _, wg, _ = p7.forward_batch(p7_in(Av_rich), vocab.unsqueeze(0), [na], [V])
        gate_base = torch.zeros(1, 12, device=DEVICE)
        # 模拟收敛gate
        abc_gate_scale = 0.05  # 收敛时ABC退出
        gate_input = gate_base + abc_gate_scale*abc_to_gate(torch.zeros(1,48,device=DEVICE)) + sent_to_gate(sv)
        gates = meta(explore(gate_input))
        logits = p6(sv, ce.weight, gate=gates)[:, :nb, :]
        preds = logits[0].argmax(-1).tolist()
        ps = ''.join([i2c.get(p,'?') for p in preds])
        ts = ''.join(B)
        ok = sum(1 for p,t in zip(preds, B) if i2c.get(p,'?') == t)
        total_ok += ok; total_n += nb
        if idx < 5:
            print(f"  [{ok}/{nb}] {ps[:50]}")
            print(f"        {ts[:50]}")

acc = total_ok/total_n*100 if total_n > 0 else 0
print(f"\n  考试准确率: {acc:.1f}% ({total_ok}/{total_n}字)")
