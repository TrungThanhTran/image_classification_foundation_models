import torch
import torchvision

"""Image classification model"""
import timm
import clip

class ClipBase32(torch.nn.Module):
    def __init__(self: None,
                 num_classes: int):
        """
        input:
        + num_classes: the number of labels for classification
        """
        super(ClipBase32, self).__init__()
        clip_model, _ = clip.load("ViT-B/32", jit=False, device='cpu')
        self.model = clip_model.visual
        self.model.proj = None

        fully_connecteds = []
        size_current = 768
        fc_last = torch.nn.Linear(size_current, num_classes)

        torch.nn.init.kaiming_uniform_(fc_last.weight, nonlinearity="relu")

        fully_connecteds.append(fc_last)

        self.classifier = torch.nn.Sequential(*fully_connecteds)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.model(x)
        f = self.classifier(f)
        return f

class ImageEncoderViT(torch.nn.Module):
    def __init__(self,
                 model_name: str = 'clip_base_32',
                 global_pool: str = 'token',
                 freeze_backbone: bool = True,
                 use_pretrain: bool = True,
                 blocks_to_freeze: list = [0]) -> None:
        super(ImageEncoderViT, self).__init__()
        # try:
        print(f'Use pretrained model: {use_pretrain}')
        print(f'Global pooling type: {global_pool}')
        print(f'Model name: {model_name}')
        
        self.model_name = model_name
        
        if self.model_name == 'vit_tiny':
            model = timm.create_model('vit_tiny_patch16_224', pretrained=use_pretrain)
        elif self.model_name == 'vit_small':
            model = timm.create_model('vit_small_patch16_224', pretrained=use_pretrain)
        elif self.model_name == 'vit_base':
            model = timm.create_model('vit_base_patch16_224', pretrained=use_pretrain)
        elif self.model_name == 'vit_base_21k':
            model = timm.create_model('vit_base_patch16_224_in21k', pretrained=use_pretrain)
        elif self.model_name == 'clip_base_32':
            clip_model, _ = clip.load("ViT-B/32", jit=False, device='cpu')
            model = clip_model.visual
            model.proj = None
        elif self.model_name == 'clip_base_16':
            clip_model, _ = clip.load("ViT-B/16", jit=False, device='cpu')
            model = clip_model.visual
            model.proj = None
        elif self.model_name == 'clip_large':
            clip_model, _ = clip.load("ViT-L/14", jit=False, device='cpu')
            model = clip_model.visual
            model.proj = None
        elif self.model_name == 'clip_large_336':
            clip_model, _ = clip.load("ViT-L/14@336px", jit=False, device='cpu')
            model = clip_model.visual
            model.proj = None

        if freeze_backbone:
            if self.model_name not in ['flava_full', 'clip_base_32', 'clip_base', 'clip_large', 'clip_large_336']:
                for block_idx in blocks_to_freeze:
                    if block_idx > len(model.blocks):
                        print('Freeze all layers!!!')
                        for _, p in model.named_parameters():
                            p.requires_grad = False
                        break
                    if block_idx < 0:
                        for _, p in model.named_parameters():
                            p.requires_grad = False
                        for p in model.norm.parameters():
                            p.requires_grad = True
                        print('Unfreeze for ', str(-block_idx), ' layers!!!')
                        for b_idx in range(len(model.blocks) + block_idx, len(model.blocks)):
                            print('Unfreeze block:', b_idx)
                            for p in model.blocks[b_idx].parameters():
                                p.requires_grad = True
                        break
                    for p in model.blocks[block_idx].parameters():
                        p.requires_grad = False
                self.embed_dim = model.embed_dim
            else:
                print('Freeze all layers!!!')
                for p in model.parameters():
                    p.requires_grad = False
                self.embed_dim = 768

        self.global_pool = global_pool
        self.model = model
        # self.model = torch.nn.Sequential(*list(model.children())[:-1])
        # for _, p in self.model.named_parameters():
        #     print(p.requires_grad)

    def forward(self, transformed_img):
        if self.global_pool == 'token':
            if self.model_name == 'flava_full':
                feat = self.model(transformed_img).last_hidden_state[:, 0, :]
            elif self.model_name in ['clip_base_32', 'clip_base', 'clip_large', 'clip_large_336']:
                feat = self.model(transformed_img)
            else:
                feat = self.model.forward_features(transformed_img)[:, 0]
        elif self.global_pool == 'avg':
            feat = self.model.forward_features(transformed_img)[:, 1:].mean(dim=1)
        # feat = self.model(transformed_img)[:, 0]
        return feat


class LinearLayer(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 num_classes: int):

        super(LinearLayer, self).__init__()
        linear = torch.nn.Linear(in_features, num_classes)
        torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")

        self.model = torch.nn.Sequential(
            linear
        )

    def forward(self, x):
        return self.model(x)


class TSModel(torch.nn.Module):
    def __init__(self,
                 model_name: str,
                 global_pool: str,
                 use_pretrain: bool = True,
                 num_classes: int = 2,
                 freeze_backbone: bool = False,
                 blocks_to_freeze: list = [1, 2, 3, 4]):
        
        super(TSModel, self).__init__()
       
        if  'clip' in model_name:
            self.backbone = ImageEncoderViT(model_name=model_name,
                                            global_pool=global_pool,
                                            use_pretrain=use_pretrain,
                                            freeze_backbone=freeze_backbone,
                                            blocks_to_freeze=blocks_to_freeze)
            self.backbone.model.head = torch.nn.Identity()
            if 'clip' in model_name:
                if model_name in ['clip_base_32', 'clip_base']:
                    embed_dim = 768
                else:
                    embed_dim = 1024
            else:
                embed_dim = self.backbone.model.embed_dim
            self.f = LinearLayer(in_features = embed_dim, num_classes = num_classes)

        # self.dropout = torch.nn.Dropout(p = 0.2)

    def forward(self, x):
        feat = self.backbone(x)
        out = self.f(feat)
        # out = self.f(self.dropout(feat))
        return out

class ResNet50(torch.nn.Module):
    def __init__(self: None, 
                 num_classes: int, 
                 use_pretrain: bool = True, 
                 blocks_to_freeze: list = [1, 2, 3, 4]) -> None:
        """
        input:
            + num_classes: the number of labels for classification
            + use_pretrain: if True, use imagenet pretrained model
            + blocks_to_freeze: resnet has 4 main block, choose the blocks
            you want to freeze 
        """
        assert len(blocks_to_freeze) <= 4
        super(ResNet50, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=use_pretrain)
        
        # Freeze blocks
        blocks = {1: resnet.layer1, 2: resnet.layer2,
                  3: resnet.layer3, 4: resnet.layer4}
        for block_index in blocks_to_freeze:
            for p in blocks[block_index].parameters():
                p.requires_grad = False

        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])

        fully_connecteds = []
        size_current = 2048
        fc_last = torch.nn.Linear(size_current, num_classes)

        torch.nn.init.kaiming_uniform_(fc_last.weight, nonlinearity="relu")

        fully_connecteds.append(fc_last)

        self.classifier = torch.nn.Sequential(*fully_connecteds)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.model(x)
        f = torch.flatten(f, 1)
        f = self.classifier(f)
        # f = self.softmax(f)
        return f  
