# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 2.2  Proposed: AAB-L — light symmetric residual attention block with a learned per-image gate router, and its
#      family of controls (one class, switches per variant). The old AAB is no longer used.
#   Mc = sigmoid(ECA-conv([dwpool4(F); std(F)] standardised over channels) + b_c)        channel mask  [B,C]
#   Ms = sigmoid(conv7x7([mean_c F; max_c F]) + b_s)                                     spatial mask  [B,1,H,W]
#   M  = a Mc + (1 - a) Ms ;  a = gate
#   modulation  sym      : F' = F (1 + gamma (2M - 1))     amplify AND suppress (AAB-L)
#               suppress : F' = F (1 - gamma (1 - M))      suppress only, same parameters (control: no amplification)
#   gate        adaptive : a = sigmoid(s + f(ep) R (tanh r(x) - batch mean)) (running mean at inference; router on
#                          [4 mask statistics, LayerNorm(Linear(2C -> 16))]; last layer zero-init; f = 0 -> 1 ep 6-10)
#               const    : the same router fed LEARNED CONSTANT inputs (+2C+4 params) -> image-independent gate
#               fixed    : a = 0.5, nothing learned (router parameters kept -> capacity >= AAB-L)
#   spatial     False    : Ms replaced by a second CHANNEL mask Mc2 (ECA conv + C biases >= the 99 params replaced)
#   gamma       sigmoid  : gamma = sigmoid(raw), init 0.7 (AAB-L as already trained)
#               ste      : gamma = clamp(raw, 0, 1), straight-through gradient -> gamma0 = 0 is reachable (exact
#                          identity at init = the saddle); used by the one-lever anti-saddle variants
#               hard     : torch.clamp (diagnostic only). Whether its gradient at gamma = 0 is zero (absorbing point)
#                          depends on the PyTorch version -> measured in 5.4, not assumed
#   Computed in factorised form F' = F + F (u + w * Ms): one full-size temporary per block. Stages 3-4 (9 blocks).
# =============================================================================
def _eca_k(c):
    t = int(abs((math.log2(c) + 1) / 2))
    return t if t % 2 else t + 1


# AAB-L defaults = AABL_CFG (cell 0.2, unchanged -> the AAB-L runs already on disk stay valid) + the variant switches
V2_DEFAULT = dict({k: AABL_CFG[k] for k in ("gamma0", "asym", "gate_a0", "router_dim", "pool_dw", "branch_drop",
                                              "route_scale", "run_momentum")},
                  mod="sym", gate="adaptive", spatial=True, gamma_param="sigmoid")


class AABL(nn.Module):
    def __init__(self, c, **kw):
        super().__init__()
        self.cfg = cfg = {**V2_DEFAULT, **kw}; self.C = c; k = _eca_k(c)
        assert cfg["mod"] in ("sym", "suppress", "v1") and cfg["gate"] in ("adaptive", "const", "fixed")
        self.pool_w = nn.Parameter(torch.full((c, 16), 1.0 / 16)) if cfg["pool_dw"] else None
        self.ch_conv = nn.Conv1d(2, 1, k, padding=k // 2, bias=False)
        self.ch_bias = nn.Parameter(torch.full((c,), float(cfg["asym"])))
        nn.init.normal_(self.ch_conv.weight, 0.0, 0.01)
        if cfg["spatial"]:
            ks = CFG.AAB_SA_KERNEL
            self.sa = nn.Conv2d(2, 1, ks, padding=ks // 2, bias=False); nn.init.normal_(self.sa.weight, 0.0, 0.01)
            self.ms_bias = nn.Parameter(torch.tensor(-float(cfg["asym"])))
        else:
            self.ch_conv2 = nn.Conv1d(2, 1, k, padding=k // 2, bias=False); nn.init.normal_(self.ch_conv2.weight, 0.0, 0.01)
            self.ch_bias2 = nn.Parameter(torch.full((c,), -float(cfg["asym"])))
        a0 = float(cfg["gate_a0"])
        self.gate_s = nn.Parameter(torch.tensor(math.log(a0 / (1 - a0))))
        d = int(cfg["router_dim"])
        self.gate_proj = nn.Linear(2 * c, d); self.gate_norm = nn.LayerNorm(d)
        self.gate_mlp = nn.Sequential(nn.Linear(4 + d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.zeros_(self.gate_mlp[-1].weight); nn.init.zeros_(self.gate_mlp[-1].bias)
        if cfg["gate"] == "const":
            self.gate_const_dn = nn.Parameter(torch.zeros(2 * c)); self.gate_const_st = nn.Parameter(torch.zeros(4))
        self.register_buffer("res_run", torch.zeros(()))
        g0 = float(cfg["gamma0"])
        if cfg["mod"] == "v1":
            self.raw_gamma = None
        elif cfg["gamma_param"] == "sigmoid":                     # AAB-L: gamma = sigmoid(raw), init 0.7
            self.raw_gamma = nn.Parameter(torch.full((c,), math.log(g0 / (1 - g0))))
        else:                                                     # "ste" / "hard": gamma = clamp(raw, 0, 1), raw = gamma0
            self.raw_gamma = nn.Parameter(torch.full((c,), g0))
        self.res_factor = 1.0                       # f(ep), set every epoch by train_one (1.0 after training)
        self.a_given = self.a_const = self.gamma_override = None      # diagnostics
        self.track = False; self.a_g = None; self.probe = False; self.probe_out = None
        self.measure = False; self.act = []         # activity ||F' - F|| / ||F|| per image
        self.last_a = self.last_a_pre = None

    def gain(self):
        if self.raw_gamma is None:
            return None
        p = self.cfg["gamma_param"]
        if p == "sigmoid":
            return torch.sigmoid(self.raw_gamma)
        g = self.raw_gamma.clamp(0.0, 1.0)
        return self.raw_gamma + (g - self.raw_gamma).detach() if p == "ste" else g      # "hard": zero grad at 0 / 1

    def _masks(self, x):
        st = torch.float32
        var, mean = torch.var_mean(x, dim=(2, 3), unbiased=False)
        pooled = ((F.adaptive_avg_pool2d(x, 4).flatten(2).to(st) * self.pool_w).sum(-1) if self.pool_w is not None
                  else mean.to(st))
        dd = torch.stack([pooled, (var.to(st) + 1e-5).sqrt()], 1)
        dn = (dd - dd.mean(2, keepdim=True)) / (dd.std(2, keepdim=True, unbiased=False) + 1e-5)
        mc = torch.sigmoid(self.ch_conv(dn).squeeze(1) + self.ch_bias)
        if self.cfg["spatial"]:
            avg = x.mean(1, keepdim=True, dtype=st); mx = x.amax(1, keepdim=True).to(st)
            ms = torch.sigmoid(self.sa(torch.cat([avg, mx], 1)) + self.ms_bias)                     # [B,1,H,W]
        else:
            ms = torch.sigmoid(self.ch_conv2(dn).squeeze(1) + self.ch_bias2)[:, :, None, None]      # [B,C,1,1]
        return mc, ms, dn

    def _route(self, mc, ms, dn):
        B = mc.shape[0]; g = self.cfg["gate"]
        if g == "fixed":
            return torch.full((B,), 0.5, device=mc.device)
        if g == "const":
            z = torch.cat([self.gate_const_st, self.gate_norm(self.gate_proj(self.gate_const_dn))]).view(1, -1)
            t = torch.tanh(self.gate_mlp(z)).view(1)
            return torch.sigmoid(self.gate_s + self.res_factor * self.cfg["route_scale"] * t).expand(B)
        msf = ms.flatten(1)
        z = torch.cat([torch.stack([mc.mean(1), mc.std(1, unbiased=False), msf.mean(1), msf.std(1, unbiased=False)], 1),
                       self.gate_norm(self.gate_proj(dn.flatten(1)))], 1)
        t = torch.tanh(self.gate_mlp(z)).squeeze(1)
        if self.training:
            m = t.mean()
            with torch.no_grad():
                self.res_run.mul_(1 - self.cfg["run_momentum"]).add_(self.cfg["run_momentum"] * m.detach())
        else:
            m = self.res_run
        return torch.sigmoid(self.gate_s + self.res_factor * self.cfg["route_scale"] * (t - m))

    def _gate(self, x):
        mc, ms, dn = self._masks(x)
        a = self._route(mc, ms, dn)
        self.last_a_pre = a.detach()
        if self.training and self.track and a.requires_grad:
            a.retain_grad(); self.a_g = a
        if self.training and self.cfg["branch_drop"] > 0 and self.cfg["gate"] != "fixed" and not self.probe:
            drop = torch.rand(a.shape[0], device=a.device) < self.cfg["branch_drop"]
            a = torch.where(drop, (torch.rand(a.shape[0], device=a.device) < 0.5).to(a.dtype), a)
        if self.a_given is not None:
            a = torch.as_tensor(self.a_given, device=x.device, dtype=torch.float32).view(-1)
        if self.a_const is not None:
            a = torch.full_like(a, float(self.a_const))
        if self.probe:
            if a.requires_grad:
                a.retain_grad()
            self.probe_out = dict(a=a, mc=mc, ms=ms)
        self.last_a = a.detach()
        g = self.gain()
        if g is not None and self.gamma_override is not None:
            g = torch.full_like(g, float(self.gamma_override))
        return mc, ms, a, g

    def _uw(self, mc, a, g):
        a2 = a[:, None]; mod = self.cfg["mod"]
        if mod == "sym":
            return g * (2 * a2 * mc - 1), 2 * g * (1 - a2)
        if mod == "suppress":
            return g * (a2 * mc - 1), g * (1 - a2)
        if self.gamma_override is not None:                      # v1 "off" = identity
            z = torch.zeros_like(mc); return z, z
        return a2 * mc - 1, (1 - a2).expand_as(mc)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            mc, ms, a, g = self._gate(x)
            u, w = self._uw(mc, a, g)
            u, w = u[:, :, None, None], w[:, :, None, None]
        dt = x.dtype
        t = torch.addcmul(u.to(dt), w.to(dt), ms.to(dt))                         # s - 1
        if self.measure:
            with torch.no_grad():
                xt = (x * t).float().flatten(1).norm(dim=1) / x.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                self.act.append(xt.cpu())
        return torch.addcmul(x, x, t)

    def forward_reference(self, x):
        """explicit M (fp32), for the self-test only"""
        with torch.autocast(x.device.type, enabled=False):
            f = x.float(); mc, ms, a, g = self._gate(f)
            M = a[:, None, None, None] * mc[:, :, None, None] + (1 - a[:, None, None, None]) * ms
            mod = self.cfg["mod"]
            s = (1 + g[None, :, None, None] * (2 * M - 1)) if mod == "sym" else \
                (1 - g[None, :, None, None] * (1 - M)) if mod == "suppress" else M
            return (f * s).to(x.dtype)


GATED = (AABL,)
gate_blocks = lambda m: [b for b in m.modules() if isinstance(b, AABL)]

# ---- the AAB-L family (proposal 3.2a). attn_adamw: AAB-L parameters with AdamW 2e-3 (lever "lr"); False -> protocol SGD
NAIVE = dict(gamma_param="ste", gamma0=0.0, asym=0.0, gate_a0=0.5)       # identity-at-init: gamma = 0, Mc = Ms, a = 0.5
VARIANTS = {
    "AABL":       dict(tier=1, v2=dict(), attn_adamw=True),                                  # AAB-L (main)
    "AABL_SUP":   dict(tier=2, v2=dict(mod="suppress"), attn_adamw=True),                    # no amplification
    "AABL_CONST": dict(tier=2, v2=dict(gate="const"), attn_adamw=True),                      # no adaptive gating
    "AABL_FIX":   dict(tier=2, v2=dict(gate="fixed"), attn_adamw=True),                      # gate fixed at 0.5
    "AABL_NOSP":  dict(tier=2, v2=dict(spatial=False), attn_adamw=True),                     # no spatial descriptor
    "AABL_NAIVE":   dict(tier=3, v2=dict(NAIVE), attn_adamw=False),                          # identity-at-init
    "AABL_N_GAMMA": dict(tier=3, v2=dict(NAIVE, gamma0=0.7), attn_adamw=False),              # + gamma init
    "AABL_N_INIT":  dict(tier=3, v2=dict(NAIVE, asym=1.0, gate_a0=0.55), attn_adamw=False),  # + mask / gate init
    "AABL_N_LR":    dict(tier=3, v2=dict(NAIVE), attn_adamw=True),                           # + AdamW 2e-3 for AAB-L
    "AABL_HARD":    dict(tier=0, v2=dict(NAIVE, gamma_param="hard"), attn_adamw=False),      # diagnostic, never trained
}
ABLATION_CONTROLS = ("AABL_SUP", "AABL_CONST", "AABL_FIX", "AABL_NOSP")
LEVER_VARIANTS = ("AABL_NAIVE", "AABL_N_GAMMA", "AABL_N_INIT", "AABL_N_LR", "AABL")
MODULE_FACTORY = {"NA": None, "SE": SE, "CBAM": CBAM, "ECA": ECA, "TA": TripletAttention}   # references (not retrained)
for _v, _sp in VARIANTS.items():
    MODULE_FACTORY[_v] = (lambda sp: (lambda c: AABL(c, **sp["v2"])))(_sp)


def impl_tag(module):
    """AAB-L keeps the tag of the runs already trained (AABL_IMPL); other family members add their switches"""
    if module == "AABL":
        return AABL_IMPL
    if module in VARIANTS:
        return AABL_IMPL + "-" + hashlib.md5(json.dumps(VARIANTS[module], sort_keys=True, default=str).encode()).hexdigest()[:6]
    return ""


print("AAB-L family:", ", ".join(f"{DISPLAY[k]} (tier {v['tier']})" for k, v in VARIANTS.items()))
