import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _scatter_softmax_temp(logits, dst_idx, N, degree):
    # applying sqrt degree temp to avoid hub nodes washing out the attention
    H = logits.size(1)
    tau = degree[dst_idx].float().clamp(min=1.0).sqrt().unsqueeze(-1)
    scaled = logits / tau

    idx = dst_idx.unsqueeze(-1).expand(-1, H)
    maxv = torch.full((N, H), float('-inf'), device=logits.device)
    maxv.scatter_reduce_(0, idx, scaled, reduce='amax', include_self=True)

    ex = torch.exp(torch.clamp(scaled - maxv[dst_idx], -50, 50))
    sumx = torch.zeros(N, H, device=logits.device)
    sumx.scatter_add_(0, idx, ex)
    return ex / (sumx[dst_idx] + 1e-10)

def _scatter_add(src, dst_idx, N):
    out = torch.zeros(N, src.size(1), src.size(2), device=src.device)
    out.scatter_add_(0, dst_idx.view(-1, 1, 1).expand_as(src), src)
    return out

class PGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_c = out_channels
        D = heads * out_channels

        self.W_src = nn.Linear(in_channels, D, bias=False)
        self.W_dst = nn.Linear(in_channels, D, bias=False)

        # kept these 2D so xavier uniform doesn't crash on us later
        self.a_src = nn.Parameter(torch.empty(heads, out_channels))
        self.a_dst = nn.Parameter(torch.empty(heads, out_channels))

        self.w_g = nn.Parameter(torch.tensor(1.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))

        self.W_out = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.a_src)         
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.constant_(self.W_out.bias, 0.0)

    def forward(self, x, edge_index, edge_attr, degree):
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        p = edge_attr.squeeze(-1)                                      

        h_s = self.W_src(x).view(N, self.heads, self.out_c)            
        h_d = self.W_dst(x).view(N, self.heads, self.out_c)            

        a_src = self.a_src.unsqueeze(0)
        a_dst = self.a_dst.unsqueeze(0)

        s = (h_s[src_idx] * a_src).sum(-1) + (h_d[dst_idx] * a_dst).sum(-1)                             

        # calculate the prob gate and intercept the logit
        g = 1.0 + torch.tanh(self.w_g * p + self.b_g)                 
        e = F.leaky_relu(g.unsqueeze(-1) * s, 0.2)                    

        alpha = _scatter_softmax_temp(e, dst_idx, N, degree)            
        alpha = self.drop(alpha)

        p_sum = torch.zeros(N, device=x.device)
        p_sum.scatter_add_(0, src_idx, p)
        p_norm = p / (p_sum[src_idx] + 1e-10)                          

        # double probability message scaling
        msg = h_s[src_idx] * (alpha * p_norm.unsqueeze(-1)).unsqueeze(-1)
        z = _scatter_add(msg, dst_idx, N).view(N, -1)                

        return self.norm(x + F.elu(self.W_out(z)))                     

class PGAT_IM(nn.Module):
    def __init__(self, in_channels=11, hidden_channels=32, embed_dim=32, heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        D = heads * hidden_channels

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList([
            PGATConv(D, hidden_channels, heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(D + in_channels, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
        )

        self.ic_head = nn.Sequential(nn.Linear(embed_dim, 16), nn.GELU(), nn.Linear(16, 1))
        self.lt_head = nn.Sequential(nn.Linear(embed_dim, 16), nn.GELU(), nn.Linear(16, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)

        # get degrees for the temp scaler, clamp to avoid div by zero
        degree = torch.zeros(N, dtype=torch.long, device=x.device)
        degree.scatter_add_(
            0, edge_index[1],
            torch.ones(edge_index.size(1), dtype=torch.long, device=x.device)
        )
        degree.clamp_(min=1)

        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr, degree)

        embed = self.output_proj(torch.cat([h, x], dim=-1))
        return embed, self.ic_head(embed), self.lt_head(embed)

class PGATLoss(nn.Module):
    def __init__(self, delta=0.1, lam=0.5):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.lam = lam

    def forward(self, y_ic_hat, y_ic, y_lt_hat, y_lt):
        # using huber to cap gradients on massive cascade outliers
        l_ic = self.huber(y_ic_hat, y_ic)
        l_lt = self.huber(y_lt_hat, y_lt)
        return l_ic + self.lam * l_lt, l_ic, l_lt

def build_optimizer(model, base_lr=1e-3, llrd=0.8, wd=1e-4, epochs=300, warmup=20):
    num_layers = len(model.convs)

    groups = [
        {
            'params': model.input_proj.parameters(),
            'lr': base_lr * (llrd ** num_layers),
        }
    ]

    # dynamic LLRD based on layer depth
    for k, conv in enumerate(model.convs):
        groups.append({
            'params': conv.parameters(),
            'lr': base_lr * (llrd ** (num_layers - 1 - k)),
        })

    groups.append({
        'params': (
            list(model.output_proj.parameters()) +
            list(model.ic_head.parameters()) +
            list(model.lt_head.parameters())
        ),
        'lr': base_lr,
    })

    opt = torch.optim.AdamW(groups, weight_decay=wd)

    def lr_fn(epoch):
        if epoch < warmup:
            return epoch / max(warmup, 1)
        t = (epoch - warmup) / max(epochs - warmup, 1)
        # using standard math.cos here to avoid tensor tracing issues
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    return opt, scheduler