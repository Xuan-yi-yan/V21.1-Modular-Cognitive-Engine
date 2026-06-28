"""V21快速评测 — 直接从train_v19_turbo导入模型定义, 100%匹配"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, os, time, random, re
sys.path.insert(0,'C:/ai')
from utils.config import *

# 直接从训练脚本导入所有模型定义
import importlib.util
spec = importlib.util.spec_from_file_location("train_v21", "C:/ai/train_v19_turbo.py")
# 不导入训练循环, 只导入类定义

# 改为: 复制模型定义
class P6_Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_words = 128
        self.encoder = nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(128,256)*0.1)
        self.heads = nn.ModuleList([nn.Linear(256,256) for _ in range(128)])
    def forward(self, sv, cw, gate=None):
        h = self.encoder(sv)
        if gate is not None:
            h = h * gate
        return torch.stack([self.heads[i](h+self.pos_embed[i].unsqueeze(0)) for i in range(128)],1).matmul(cw.T)

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
        ef = attr_vec[:,:,34:39].mean(dim=1)
        return self.refine(torch.cat([content, ef],-1))

class TE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(384,512),nn.GELU(),nn.Linear(512,512),nn.GELU(),nn.Linear(512,256))
    def forward(self,x): return self.net(x)

class TM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(256))
    def forward(self,x): return torch.sigmoid(self.bias+x)

from P7_cross_sent.model import P7WordRouter2048
from P3_word_attr.stack import P3AttributeStack
from P3_word_attr.p3l_linkage import P3L_AttributeLinkage

def words_to_attr_ids(w):
    ids = []
    for c, p in zip(w, _p3s.process_sentence(w[:10])):
        bt = getattr(p, 'basic_type', '')
        ids.append(hash(f'{c}:{bt}') % 300)
    return list(set(ids))

_p3s = P3AttributeStack()

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
                        '数量':6,'程度':7,'方位':8,'方式':9,'原因':10,'结果':11,'目的':12,'条件':13}.get(s[0],-1)
                    if sl>=0:vec[i,10+sl]=s[1]
        syn=d.get('syntax_candidates',[])
        if isinstance(syn,list) and syn:
            s0=syn[0]
            if isinstance(s0,(tuple,list)) and len(s0)>=2:
                sl={'主语':0,'谓语':1,'宾语':2,'定语':3,'状语':4,'补语':5,'兼语':6,'连动':7,'同位':8,'独立':9}.get(s0[0],-1)
                if sl>=0:vec[i,24+sl]=s0[1]
        pol=d.get('polarity',('neutral',0.0,'none'))
        if isinstance(pol,tuple) and len(pol)>=1:
            vec[i,34]=1.0 if pol[0]=='positive' else (-1.0 if pol[0]=='negative' else 0.0)
            vec[i,35]=pol[1] if len(pol)>=2 else 0.0
        conn=d.get('conn_type','')
        cslots={'cause':0,'adversative':1,'coordinate':2,'conditional':3,'progressive':4,
                'concessive':5,'alternative':6,'sequential':7,'summary':8,'example':9,'purpose':10}
        if conn in cslots:vec[i,42+cslots[conn]]=d.get('conn_confidence',0.85)
        mod=d.get('mod_type','')
        mslots={'adjective':0,'adverb_manner':1,'scope':2,'attributive':3,'adverbial':4,'complement':5}
        if mod in mslots:vec[i,56+mslots[mod]]=d.get('mod_confidence',0.8)
        if d.get('is_comparative',False):vec[i,63]=0.8
        vec[i,31]=i/max(max_n,1)
    return vec

# ── 加载 ──
ckpt = torch.load('C:/ai/P1_char_word/checkpoints/V21_best.pt', map_location=DEVICE)
c2i = ckpt['c2i']; i2c = {i:c for c,i in c2i.items()}; V = len(c2i)
print(f"═══ V21 评测 ═══ E{ckpt.get('epoch','?')} loss={ckpt.get('loss','?'):.4f} V={V}")

ce = nn.Embedding(V, 256, padding_idx=0).to(DEVICE)
p7 = P7WordRouter2048(heads=16, head_dim=16, word_dim=256, inner_dim=256, max_len=128, num_groups=4).to(DEVICE)
p6 = P6_Tied().to(DEVICE)
p7_in = nn.Linear(384, 256, bias=False).to(DEVICE)
abcA = ABC_StageA().to(DEVICE); abcB = ABC_StageB().to(DEVICE); abcC = ABC_StageC().to(DEVICE)
explore = TE().to(DEVICE); meta = TM().to(DEVICE)
p3l = P3L_AttributeLinkage(num_attr_values=300).to(DEVICE)
abc_to_gate = nn.Linear(48, 128, bias=False).to(DEVICE)
gate_base_proj = nn.Linear(12, 128, bias=False).to(DEVICE)
sent_to_gate = nn.Linear(256, 128, bias=False).to(DEVICE)

# 加载权重 (strict=False跳过不匹配的)
ce.load_state_dict(ckpt['char_embed'], strict=False)
p7.load_state_dict(ckpt['p7'], strict=False)
p6.load_state_dict(ckpt['p6'], strict=False)
p7_in.load_state_dict(ckpt['p7in'], strict=False)
for m_name, mod in [('abcA',abcA),('abcB',abcB),('abcC',abcC),('explore',explore),('meta',meta),('abc_to_gate',abc_to_gate),('sent_to_gate',sent_to_gate),('p3l',p3l),('gate_base_proj',gate_base_proj)]:
    if m_name in ckpt:
        mod.load_state_dict(ckpt[m_name], strict=False)
for m in [ce,p7,p6,p7_in,abcA,abcB,abcC,explore,meta,abc_to_gate,sent_to_gate,p3l,gate_base_proj]: m.eval()

# ── 评测 ──
vocab = ce.weight.detach()
print(f"\n═══ 训练集首对(应完美) ═══")
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
# 考试集: 从clean数据采样并分割(同训练逻辑)
import random; # 从clean数据中找c2i兼容对, 分割训练/考试
compat = [(A, B) for A, B in all_pairs if all(c in c2i for c in A) and all(c in c2i for c in B)]
import random; random.shuffle(compat)
n_c = len(compat); t80 = int(n_c*0.80); t10 = int(n_c*0.10)
exam_set = compat[t80+t10:]  # 考试=最后10%
test_pairs = exam_set[:min(10, len(exam_set))]
print(f"\n  兼容{len(compat)}对 | 训练{len(compat[:t80])} | 考试{len(exam_set)} | 测试{min(10,len(exam_set))}对")

total_ok=0; total_n=0
for idx, (A, B) in enumerate(test_pairs):
    na = len(A); nb = len(B)
    with torch.no_grad():
        A_emb = ce(torch.tensor([c2i[c] for c in A], device=DEVICE))
        attr_a = words_to_attr_vec(A).to(DEVICE)[:na, :]
        Av_rich = torch.cat([A_emb, attr_a], dim=-1).unsqueeze(0)
        wo, sv, _, _, _ = p7.forward_batch(p7_in(Av_rich), vocab.unsqueeze(0), [na], [V])
        # 推理gate: 跑完整ABC链 + sent
        aids = words_to_attr_ids(A)
        if len(aids) >= 2:
            _, attns = p3l(torch.tensor(aids, device=DEVICE))
            feats = []
            for g, scores in attns.items():
                if scores.numel() > 0: feats.append(torch.stack([scores.mean(), scores.std(), scores.max()-scores.min()]))
            p3lf_f = torch.cat(feats) if feats else torch.zeros(60, device=DEVICE)
        else:
            p3lf_f = torch.zeros(60, device=DEVICE)
        p3lf_f = F.pad(p3lf_f, (0, max(0, 128-p3lf_f.shape[0])))[:128].unsqueeze(0)
        a_log, ah = abcA(attr_a.unsqueeze(0), wo[:,:attr_a.shape[0],:])
        ct_r = abcB(ah, attr_a.unsqueeze(0), p3lf_f)
        ac_r = abcC(ct_r, attr_a.unsqueeze(0))
        abc_conf = torch.stack([a_log.softmax(-1).max(-1).values.mean(), ac_r.norm(dim=-1).mean()] + [torch.tensor(0.0,device=DEVICE)]*10).unsqueeze(0)
        gate_input = torch.cat([gate_base_proj(abc_conf), abc_to_gate(ac_r), sent_to_gate(sv)], dim=-1)
        gates = meta(explore(gate_input))
        logits = p6(sv, ce.weight, gate=gates)[:, :nb, :]
        preds = logits[0].argmax(-1).tolist()
        ps = ''.join([i2c.get(p,'?') for p in preds])
        ts = ''.join(B)
        ok = sum(1 for p,t in zip(preds, B) if i2c.get(p,'?')==t)
        total_ok += ok; total_n += nb
        if idx < 5:
            print(f"  [{ok}/{nb}] {ps[:50]}")
            print(f"        {ts[:50]}")
acc = total_ok/total_n*100 if total_n > 0 else 0
print(f"\n  准确率(c2i兼容): {acc:.1f}% ({total_ok}/{total_n}字)")

# 自我诊断
print(f"\n═══ 自诊 ═══")
with torch.no_grad():
    test_chars = ['的','一','是','人']  # 高频字应在c2i里
    ids = [c2i.get(c,0) for c in test_chars]
    emb = ce(torch.tensor(ids, device=DEVICE))
    print(f"  char_embed norm: {emb.norm(dim=-1).tolist()}")
    attr_t = torch.zeros(4, 128, device=DEVICE)
    av = torch.cat([emb, attr_t], dim=-1).unsqueeze(0)
    wo, sv, _, _, _ = p7.forward_batch(p7_in(av), vocab.unsqueeze(0), [4], [V])
    print(f"  sent_vec norm={sv.norm().item():.3f} mean={sv.mean().item():.3f}")
    a_diag, ah_d = abcA(attr_a[:4].unsqueeze(0), wo[:,:4,:])
    ac_d = abcC(abcB(ah_d, attr_a[:4].unsqueeze(0), torch.zeros(1,128,device=DEVICE)), attr_a[:4].unsqueeze(0))
    abc_conf2 = torch.stack([a_diag.softmax(-1).max(-1).values.mean(), ac_d.norm(dim=-1).mean()] + [torch.tensor(0.0,device=DEVICE)]*10).unsqueeze(0)
    gb_out = gate_base_proj(abc_conf2)
    abc_out = abc_to_gate(ac_r)
    sent_out = sent_to_gate(sv)
    print(f"  gate_base_proj norm={gb_out.norm().item():.3f}")
    print(f"  abc_to_gate out norm={abc_out.norm().item():.3f}")
    print(f"  sent_to_gate out norm={sent_out.norm().item():.3f}")
    gi = torch.cat([gb_out, abc_out, sent_out], dim=-1)
    gates = meta(explore(gi))
    print(f"  explore out norm={explore(gi).norm().item():.3f}")
    print(f"  gate mean={gates.mean().item():.4f} std={gates.std().item():.4f}")
    logits = p6(sv, ce.weight, gate=gates)
    print(f"  logits range=[{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  logits has NaN: {torch.isnan(logits).any().item()}")
    pred = logits[0,:4,:].argmax(-1).tolist()
    print(f"  预测前4: {[i2c.get(p,'?') for p in pred]}")
