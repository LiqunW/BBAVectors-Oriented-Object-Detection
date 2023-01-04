import torch
try:
    from .RepVGG.repvgg import get_RepVGG_func_by_name
except:
    from RepVGG.repvgg import get_RepVGG_func_by_name


class RepVGG(torch.nn.Module):
    def __init__(self, backbone_name, backbone_file="", pretrained=False, deploy=False):
        super(RepVGG, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt, strict=False)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, \
                                                                          backbone.stage2, backbone.stage3, \
                                                                          backbone.stage4

    def forward(self, x):
        feat = []
        feat.append(x)
        x = self.layer0(x)
        feat.append(x)
        for block in self.layer1:
            x = block(x)
        feat.append(x)
        for block in self.layer2:
            x = block(x)
        feat.append(x)
        for block in self.layer3:
            x = block(x)
        feat.append(x)
        for block in self.layer4:
            x = block(x)
        feat.append(x)
        return feat


if __name__ == "__main__":
    model = RepVGG("RepVGG-B3")

    input = torch.rand(4, 3, 608, 608)
    # model.eval()
    res = model(input)
