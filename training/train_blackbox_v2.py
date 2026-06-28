"""
黑盒V2 — 多元组架构
可训练N元字符组 + 组级交叉注意力 + 3层GPU缓存 + 对错对比loss
"""
import torch, torch.nn as nn, torch.nn.functional as F, os, sys, random, time, argparse, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.config import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--data", type=str, default="public")
parser.add_argument("--batch_size", type=int, default=28)
parser.add_argument("--display", type=int, default=10)
parser.add_argument("--sample", type=int, default=15)
args = parser.parse_args()

w = lambda s: print(s, flush=True)

# ════════════════════════ 1. 数据 ════════════════════════
w(f"═══ 黑盒V2 多元组架构 ═══")
SENT_SPLIT = re.compile(r'[。！？；\n]')
all_pairs = []
path = "C:/ai/data/public_clean.txt" if args.data == "public" else args.data
is_cover = ('cover' in path)
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        a, b = line.split('\t', 1)
        if is_cover:
            wa = [c for c in a if c.strip() and ord(c)>32]
            wb = [c for c in b if c.strip() and ord(c)>32]
            if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                all_pairs.append((wa, wb))
        else:
            sa = [p.strip() for p in SENT_SPLIT.split(a) if len(p.strip())>=3]
            sb = [p.strip() for p in SENT_SPLIT.split(b) if len(p.strip())>=3]
            for xa, xb in zip(sa, sb):
                wa = [c for c in xa if c.strip() and ord(c)>32]
                wb = [c for c in xb if c.strip() and ord(c)>32]
                if 3<=len(wa)<=80 and 3<=len(wb)<=80:
                    all_pairs.append((wa, wb))

if args.sample > 0 and args.sample < len(all_pairs):
    all_pairs = random.sample(all_pairs, args.sample)

all_chars = set()
for A, B in all_pairs: all_chars.update(A); all_chars.update(B)
c2i = {c: i for i, c in enumerate(sorted(all_chars), start=1)}
c2i['<unk>'] = 0; c2i['<pad>'] = -1
i2c = {i:c for c,i in c2i.items()}
V = len(c2i); n_pairs = len(all_pairs)
w(f"  字表={V} | 数据={n_pairs}对")

random.shuffle(all_pairs)
n = len(all_pairs); t80 = int(n*0.80); t10 = int(n*0.10)
train_set = all_pairs[:t80]; test_set = all_pairs[t80:t80+t10]; exam_set = all_pairs[t80+t10:]
w(f"  训练:{len(train_set)} 测试:{len(test_set)} 考试:{len(exam_set)}")

# ════════════════════════ 2. 模型 ════════════════════════
G = 256        # 组数
group_dim = 64  # 每组特征维度
K_max = 12      # 每字最多归属组数

# CharEmbed
char_embed = nn.Embedding(V, 256, padding_idx=0).to(DEVICE)
nn.init.orthogonal_(char_embed.weight)

# 多元组分配: W[V, G] — 每字→256组的亲和度
group_affinity = nn.Parameter(torch.randn(V, G, device=DEVICE) * 0.02)
group_T = nn.Parameter(torch.tensor(1.0, device=DEVICE))
group_embed = nn.Parameter(torch.randn(G, group_dim, device=DEVICE) * 0.02)

# 组级交叉注意力 (仿P7, RMSNorm Q/K)
class GroupCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(group_dim, 64, bias=False)
        self.k_proj = nn.Linear(256, 64, bias=False)
        self.q_norm = nn.RMSNorm(64)
        self.k_norm = nn.RMSNorm(64)
        self.out_proj = nn.Linear(256, 256, bias=False)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.5)
    def forward(self, group_emb, vocab):
        q = self.q_norm(self.q_proj(group_emb))  # [G, 64]
        k = self.k_norm(self.k_proj(vocab))      # [V, 64]
        scores = torch.matmul(q, k.T) / math.sqrt(64)  # [G, V]
        attn = F.softmax(scores, dim=-1)
        out = self.out_proj(torch.matmul(attn, vocab))  # [G, 256]
        sent_vec = out.mean(dim=0)  # [256]
        return out, sent_vec, attn

group_attn = GroupCrossAttention().to(DEVICE)

# Gate: 组特征→调制信号
class GroupGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 256))
    def forward(self, x):
        return torch.sigmoid(self.net(x))

group_gate = GroupGate().to(DEVICE)

# P6解码 (同V21)
class P6_Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_words = 128
        self.encoder = nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.pos_embed = nn.Parameter(torch.randn(128,256)*0.1)
        self.heads = nn.ModuleList([nn.Linear(256,256) for _ in range(128)])
    def forward(self, sv, cw, gate=None):
        h = self.encoder(sv)
        if gate is not None: h = h * gate
        return torch.stack([self.heads[i](h+self.pos_embed[i].unsqueeze(0)) for i in range(128)],1).matmul(cw.T)

p6 = P6_Tied().to(DEVICE)

n_all = sum(p.numel() for m in [char_embed, group_attn, group_gate, p6] for p in m.parameters())
n_all += sum(p.numel() for p in [group_affinity, group_embed, group_T])
w(f"  参数: {n_all/1e6:.2f}M")

# ════════════════════════ 3. 优化器(分层lr) ════════════════════════
opt_embed = torch.optim.Adam(list(char_embed.parameters())+[group_affinity, group_embed, group_T], lr=args.lr*0.3)
opt_attn  = torch.optim.Adam(group_attn.parameters(), lr=args.lr*0.5)
opt_gate  = torch.optim.Adam(group_gate.parameters(), lr=args.lr*1.0)
opt_p6    = torch.optim.Adam(p6.parameters(), lr=args.lr*0.3)

# ════════════════════════ 4. 3层缓存 ════════════════════════
cache_L1 = torch.zeros(G, V, device=DEVICE)  # 当前batch
cache_L2 = torch.zeros(G, V, device=DEVICE)  # EMA
cache_L3 = torch.zeros(G, V, device=DEVICE)  # 长期

# ════════════════════════ 5. 训练 ════════════════════════
best_loss = float('inf'); best_epoch = 0
BATCH_SIZE = args.batch_size

for ep in range(1, args.epochs+1):
    random.shuffle(train_set)
    remaining = train_set[:]
    t0 = time.time()
    total_loss, total_ce, total_contrast, n = 0.0, 0.0, 0.0, 0

    while remaining:
        batch = remaining[:BATCH_SIZE]
        remaining = remaining[BATCH_SIZE:]
        bs = len(batch)

        max_len_a = max(len(x[0]) for x in batch)
        max_len_b = max(len(x[1]) for x in batch)

        Av_ids = torch.zeros(bs, max_len_a, dtype=torch.long, device=DEVICE)
        Bv_ids = torch.zeros(bs, max_len_b, dtype=torch.long, device=DEVICE)
        Aw_ids = torch.zeros(bs, max_len_a, dtype=torch.long, device=DEVICE)  # 错字ID用于对比loss
        for i, (A, B) in enumerate(batch):
            Av_ids[i, :len(A)] = torch.tensor([c2i.get(c,0) for c in A], device=DEVICE)
            Bv_ids[i, :len(B)] = torch.tensor([c2i.get(c,0) for c in B], device=DEVICE)
            Aw_ids[i, :len(A)] = torch.tensor([c2i.get(c,0) for c in A], device=DEVICE)

        # ── 前向 ──
        A_emb = char_embed(Av_ids)  # [bs, nA, 256]
        vocab = char_embed.weight.detach()

        # 多元组分配: 每字→组权重
        T_val = F.softplus(group_T) + 0.1
        # 取每个A字的组亲和度 → softmax选择
        # group_affinity: [V, G] → 取batch中字的行
        batch_aff = group_affinity[Av_ids.view(-1)].view(bs, max_len_a, G)  # [bs, nA, G]
        batch_group_w = F.softmax(batch_aff / T_val, dim=-1)  # [bs, nA, G]

        # 输入驱动组表示: 每字激活组权重 × 字嵌入 → 每组看到什么字
        group_feat = torch.bmm(batch_group_w.transpose(1,2), A_emb)  # [bs, G, 256]
        group_feat = group_feat / (batch_group_w.sum(dim=1, keepdim=True).transpose(1,2) + 1e-8)
        # 组表示=输入特征(192D) + 固定组嵌入(64D) → 投影到64D
        group_input = group_embed.unsqueeze(0) + group_feat[:, :, :64] * 0.5  # 平衡版

        g_out, _, attn_all = group_attn(group_input.reshape(-1, 64), vocab)
        group_out = g_out.reshape(bs, G, 256)  # [bs, G, 256]
        # 不平均! 取top-16组拼接 → 16×256=4096D → 线性压缩到256D
        top_k_vals, top_k_idx = group_out.norm(dim=-1).topk(16, dim=1)
        top_groups = torch.gather(group_out, 1, top_k_idx.unsqueeze(-1).expand(-1,-1,256))
        sent_vec = top_groups.reshape(bs, -1)[:, :256]  # [bs, 256]

        # 3层缓存更新
        with torch.no_grad():
            attn_batch = attn_all.reshape(bs, G, V).mean(dim=0).detach()  # [G, V]
            cache_L1 = attn_batch
            cache_L2 = 0.9 * cache_L2 + 0.1 * attn_batch
            if ep % 10 == 0:
                cache_L3 = 0.99 * cache_L3 + 0.01 * attn_batch
            cache_mix = 0.5 * cache_L1 + 0.3 * cache_L2 + 0.2 * cache_L3

        # Gate信号: 组活跃度+组多样性
        group_active = (attn_all.sum(dim=-1) > 0.01).float().mean()  # 活跃组比例
        group_conf = attn_all.max(dim=-1).values.mean()  # 组置信度
        gs_raw = group_out.mean(dim=(0,1)).detach()
        gs = torch.cat([gs_raw[:254], torch.tensor([group_active, group_conf], device=DEVICE)]).unsqueeze(0)  # [1, 256]
        gate = group_gate(gs)  # [1, 256]

        # P6解码
        logits = p6(sent_vec, char_embed.weight, gate=gate.expand(bs, -1))[:, :max_len_b, :]

        # ── Loss ──
        ce_loss = 0.0; contrast_loss = 0.0
        for i in range(bs):
            nB = min(len(batch[i][1]), max_len_b)
            tgt = torch.tensor([c2i.get(c,0) for c in batch[i][1][:nB]], device=DEVICE)
            wrong = torch.tensor([c2i.get(c,0) for c in batch[i][0][:nB]], device=DEVICE)
            ce_loss += F.cross_entropy(logits[i, :nB, :], tgt, ignore_index=0)
            # 对比loss: 惩罚输出错字
            wrong_prob = logits[i, :nB, :].softmax(-1).gather(-1, wrong.unsqueeze(-1)).squeeze(-1).mean()
            contrast_loss += wrong_prob

        ce_loss = ce_loss / bs
        contrast_loss = contrast_loss / bs
        # 组多样性loss: 防组坍缩, 直通group_affinity梯度
        batch_group_w = F.softmax(batch_aff / T_val, dim=-1)  # [bs, nA, G]
        group_usage = batch_group_w.mean(dim=(0,1))  # [G] 每组使用率
        group_entropy = -(group_usage * torch.log(group_usage + 1e-8)).sum()
        group_loss = -group_entropy * 0.1  # 最大化组熵 → 多样化组分配
        batch_loss = ce_loss + 0.1 * contrast_loss + group_loss

        # ── 优化 ──
        for o in [opt_embed, opt_attn, opt_gate, opt_p6]: o.zero_grad()
        batch_loss.backward()

        # NaN回滚
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in char_embed.parameters())
        if has_nan:
            w(" ⚡梯度NaN! 跳过")
            for o in [opt_embed, opt_attn, opt_gate, opt_p6]: o.zero_grad()
        else:
            for o in [opt_embed, opt_attn, opt_gate, opt_p6]: o.step()

        total_loss += batch_loss.item(); total_ce += ce_loss.item()
        total_contrast += contrast_loss.item(); n += 1

    # ── Epoch总结 ──
    avg_loss = total_loss/max(n,1)
    if avg_loss < best_loss:
        best_loss = avg_loss; best_epoch = ep
        torch.save({'char_embed':char_embed.state_dict(),'group_attn':group_attn.state_dict(),
                    'group_gate':group_gate.state_dict(),'p6':p6.state_dict(),
                    'group_affinity':group_affinity,'group_embed':group_embed,'group_T':group_T,
                    'c2i':c2i,'epoch':ep,'loss':best_loss},
                   "C:/ai/P1_char_word/checkpoints/BB_best.pt")

    if ep <= 3 or ep%args.display==0 or ep==args.epochs:
        A_demo, B_demo = train_set[0]
        with torch.no_grad():
            A_ids_d = torch.tensor([c2i.get(c,0) for c in A_demo], device=DEVICE).unsqueeze(0)
            A_emb_d = char_embed(A_ids_d)
            batch_aff_d = group_affinity[A_ids_d.view(-1)].view(1, -1, G)
            batch_group_w_d = F.softmax(batch_aff_d / (F.softplus(group_T)+0.1), dim=-1)
            group_feat_d = torch.bmm(batch_group_w_d.transpose(1,2), A_emb_d)
            group_feat_d = group_feat_d / (batch_group_w_d.sum(dim=1,keepdim=True).transpose(1,2)+1e-8)
            group_input_d = group_feat_d[:,:,:64] + group_embed.unsqueeze(0) * 0.1
            g_out_d, _, _ = group_attn(group_input_d.reshape(-1,64), vocab)
            g_out_d2 = g_out_d.reshape(1, G, 256)
            _, tk_idx_d = g_out_d2.norm(dim=-1).topk(16, dim=1)
            sv_d = torch.gather(g_out_d2, 1, tk_idx_d.unsqueeze(-1).expand(-1,-1,256)).reshape(1, -1)[:, :256]
            gate_d = group_gate(torch.zeros(1,256,device=DEVICE))
            logits_d = p6(sv_d, char_embed.weight, gate=gate_d)[:, :len(B_demo), :]
            preds = logits_d[0].argmax(-1).tolist()
            ps = ''.join([i2c.get(p,'?') for p in preds[:30]])
            ok = sum(1 for p,t in zip(preds, B_demo) if i2c.get(p,'?')==t)
        w(f"E{ep:4d} loss={avg_loss:.4f} CE={total_ce/max(n,1):.3f} contrast={total_contrast/max(n,1):.4f} ok={ok}/{len(B_demo)} | {time.time()-t0:.0f}s")
        w(f"  预测: {ps}")
        w(f"  正确: {''.join(B_demo[:30])}")

        # 5项健康指标
        with torch.no_grad():
            w_act = (group_affinity.abs() > 0.01).float().mean().item()
            w_update = sum(p.grad.norm().item() for p in group_attn.parameters() if p.grad is not None) / max(sum(p.norm().item() for p in group_attn.parameters()), 1e-8)
            dim_act = (group_embed.abs().mean(dim=-1) > 0.01).float().mean().item()
            w(f"  健康: 权重激活={w_act:.1%} 更新率={w_update:.4%} 维度激活={dim_act:.1%}")

w(f"\n[完成] best E{best_epoch} loss={best_loss:.4f}")
