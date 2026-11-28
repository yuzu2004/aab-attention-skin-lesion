# Step of the AAB-L pipeline: uses names defined by the earlier step files. Run through src/run.py.
# =============================================================================
# 2.3  ResNet with a module at the end of every Basic / Bottleneck block
#      ImageNet weights come ONLY from local .pth files (cell 2.4); nothing is downloaded.
# =============================================================================
class BlockWithAttention(nn.Module):
    """torchvision Basic/Bottleneck block; attention after the last BN, before the residual sum."""

    def __init__(self, block, attn):
        super().__init__()
        self.block, self.attn = block, attn

    def forward(self, x):
        b, idt = self.block, x
        out = b.relu(b.bn1(b.conv1(x)))
        if isinstance(b, Bottleneck):
            out = b.relu(b.bn2(b.conv2(out)))
            out = b.bn3(b.conv3(out))
        else:
            out = b.bn2(b.conv2(out))
        out = self.attn(out)
        if b.downsample is not None:
            idt = b.downsample(x)
        return b.relu(out + idt)


BACKBONES = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50,
             "resnet101": resnet101, "resnet152": resnet152}
LOCAL_SD = {}                       # filled by cell 2.4: backbone name -> ImageNet state_dict (1000-class fc)


def build_model(backbone, module, pretrained, seed):
    with _BUILD_LOCK:                                   # seeded construction must not interleave across threads
        return _build_model(backbone, module, pretrained, seed)


def _build_model(backbone, module, pretrained, seed):
    set_seed(seed)
    net = BACKBONES[backbone](weights=None)
    if pretrained:
        if backbone not in LOCAL_SD:
            raise RuntimeError(f"no local ImageNet weights for {backbone} (cell 2.4); nothing is downloaded")
        net.load_state_dict(LOCAL_SD[backbone], strict=True)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    cls = MODULE_FACTORY[module]
    stages = AABL_CFG["stages"] if module in VARIANTS else CFG.ATTN_LAYERS        # AAB-L family: stages 3-4
    if cls is not None:
        for li, layer in enumerate((net.layer1, net.layer2, net.layer3, net.layer4), start=1):
            if li not in stages:
                continue
            for i, blk in enumerate(layer):
                c = blk.bn3.num_features if isinstance(blk, Bottleneck) else blk.bn2.num_features
                layer[i] = BlockWithAttention(blk, cls(c))
    return net.to(DEVICE).to(memory_format=MEMFMT)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
