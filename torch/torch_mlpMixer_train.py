


if __name__ == "__main__":
    from torchsummary import summary
    import pdb
    import torch
    import numpy as np
    from util.helper_mixerMlp import MLPMixer

    # --------- base_model_param ---------
    in_channels = 3
    hidden_size = 512
    num_classes = 1000
    patch_size = 16
    resolution = 224
    number_of_layers = 8
    token_dim = 256
    channel_dim = 2048
    # ------------------------------------

    model = MLPMixer(
        in_channels=in_channels,
        dim=hidden_size,
        num_classes=num_classes,
        patch_size=patch_size,
        image_size=resolution,
        depth=number_of_layers,
        token_dim=token_dim,
        channel_dim=channel_dim
    )
    img = torch.rand(2, 3, 224, 224)
    output = model(img)
    print(output.shape)

    # summary(model, input_size=(3, 224, 224), device='cpu')
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000  # 18.528 million
    print('Trainable Parameters: %.3fM' % parameters)