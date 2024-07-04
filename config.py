# Set to "True" while testing
test = True

if test:
    root_path = 'preprocessed_datasets_test'
    model_configs = [
        'microsoft/resnet-18',  # model.classifier[-1]
        'microsoft/resnet-50',  # model.classifier[-1]
        'facebook/convnextv2-tiny-1k-224',  # model.classifier
        'microsoft/swin-tiny-patch4-window7-224',  # model.classifier
        'google/mobilenet_v2_1.0_224',  # model.classifier
        'google/vit-base-patch16-224',  # model.classifier
        'google/efficientnet-b0',  # model.classifier
        'microsoft/beit-base-patch16-224',  # model.classifier
    ]

    # model_configs = [
    #     'microsoft/resnet-18',  # model.classifier[-1]
    #     'microsoft/resnet-50',  # model.classifier[-1]
    #     'facebook/convnextv2-tiny-1k-224',  # model.classifier
    #     'microsoft/swin-tiny-patch4-window7-224',  # model.classifier
    #     # 'nateraw/vit-age-classifier',  # model.classifier
    #     # 'nvidia/mit-b0',  # model.classifier
    #     'google/mobilenet_v2_1.0_224',  # model.classifier
    #     'google/vit-base-patch16-224',  # model.classifier
    #     'microsoft/beit-base-patch16-224',  # model.classifier
    #     # 'apple/mobilevit-small',  # model.classifier
    #     # 'facebook/convnextv2-tiny-22k-384',  # model.classifier
    #     # 'google/mobilenet_v1_0.75_192',  # model.classifier
    #     # 'google/vit-base-patch16-384',  # model.classifier
    #     # 'google/vit-large-patch32-384',  # model.classifier
    #     # 'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
    #     # 'microsoft/dit-base-finetuned-rvlcdip',  # model.classifier
    #     # 'microsoft/swin-base-patch4-window7-224-in22k',  # model.classifier
    #     # 'nvidia/mit-b2'  # model.classifier
    # ]

    # model_configs = ['apple/mobilevit-small',  # model.classifier
    #                  'facebook/convnextv2-tiny-1k-224',  # model.classifier
    #                  'facebook/convnextv2-tiny-22k-384',  # model.classifier
    #                  'google/mobilenet_v1_0.75_192',  # model.classifier
    #                  'google/mobilenet_v2_1.0_224',  # model.classifier
    #                  'google/vit-base-patch16-224',  # model.classifier
    #                  # 'google/vit-base-patch16-384',  # model.classifier
    #                  # 'google/vit-large-patch32-384',  # model.classifier
    #                  'microsoft/beit-base-patch16-224',  # model.classifier
    #                  'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
    #                  'microsoft/dit-base-finetuned-rvlcdip',  # model.classifier
    #                  'microsoft/resnet-18',  # model.classifier[-1]
    #                  'microsoft/resnet-50',  # model.classifier[-1]
    #                  'microsoft/swin-base-patch4-window7-224-in22k',  # model.classifier
    #                  'microsoft/swin-tiny-patch4-window7-224',  # model.classifier
    #                  'nateraw/vit-age-classifier',  # model.classifier
    #                  'nvidia/mit-b0',  # mode3l.classifier
    #                  'nvidia/mit-b2'  # model.classifier
    #                  ]

    dataset_configs = [
        {
            'name': 'cifar10',
            'image_key': 'img',
            'label_key': 'label',
            'num_rows': 50000
        },
        # {
        #     'name': 'cifar100',
        #     'image_key': 'img',
        #     'label_key': 'fine_label',
        #     'num_rows': 50000
        # },
    ]
    chunk_size = 10
else:
    root_path = 'preprocessed_datasets'
    model_configs = ['apple/mobilevit-small',  # model.classifier
                     'facebook/convnextv2-tiny-1k-224',  # model.classifier
                     'facebook/convnextv2-tiny-22k-384',  # model.classifier
                     'google/mobilenet_v1_0.75_192',  # model.classifier
                     'google/mobilenet_v2_1.0_224',  # model.classifier
                     'google/vit-base-patch16-224',  # model.classifier
                     # 'google/vit-base-patch16-384',  # model.classifier
                     # 'google/vit-large-patch32-384',  # model.classifier
                     'microsoft/beit-base-patch16-224',  # model.classifier
                     'microsoft/beit-base-patch16-224-pt22k-ft22k',  # model.classifier
                     'microsoft/dit-base-finetuned-rvlcdip',  # model.classifier
                     'microsoft/resnet-18',  # model.classifier[-1]
                     'microsoft/resnet-50',  # model.classifier[-1]
                     'microsoft/swin-base-patch4-window7-224-in22k',  # model.classifier
                     'microsoft/swin-tiny-patch4-window7-224',  # model.classifier
                     'nateraw/vit-age-classifier',  # model.classifier
                     'nvidia/mit-b0',  # mode3l.classifier
                     'nvidia/mit-b2'  # model.classifier
                     ]
    dataset_configs = [
        {
            'name': 'cifar10',
            'image_key': 'img',
            'label_key': 'label',
            'num_rows': 50000
        },
        {
            'name': 'cifar100',
            'image_key': 'img',
            'label_key': 'fine_label',
            'num_rows': 50000
        },
        {
            'name': 'beans',
            'image_key': 'image',
            'label_key': 'labels',
            'num_rows': 1034
        },
        {
            'name': 'Matthijs/snacks',
            'image_key': 'image',
            'label_key': 'label',
            'num_rows': 4838
        },
        {
            'name': 'sasha/dog-food',
            'image_key': 'image',
            'label_key': 'label',
            'num_rows': 2100
        },
        {
            'name': 'nelorth/oxford-flowers',
            'image_key': 'image',
            'label_key': 'label',
            'num_rows': 7169
        },
        {
            'name': 'cats_vs_dogs',
            'image_key': 'image',
            'label_key': 'labels',
            'num_rows': 23410
        }
    ]
    chunk_size = 1000

model_configs_total = ['Zetatech/pvt-tiny-224', 'akahana/vit-base-cats-vs-dogs', 'apple/mobilevit-small',
                       'apple/mobilevit-x-small',
                       'apple/mobilevit-xx-small', 'deepmind/vision-perceiver-conv',
                       'deepmind/vision-perceiver-fourier',
                       'deepmind/vision-perceiver-learned', 'facebook/convnext-base-224',
                       'facebook/convnext-base-224-22k',
                       'facebook/convnext-base-384-22k-1k', 'facebook/convnext-large-224-22k-1k',
                       'facebook/convnext-large-384',
                       'facebook/convnext-tiny-224', 'facebook/convnext-xlarge-384-22k-1k',
                       'facebook/convnextv2-base-1k-224',
                       'facebook/convnextv2-base-22k-224', 'facebook/convnextv2-large-22k-224',
                       'facebook/convnextv2-large-22k-384',
                       'facebook/convnextv2-tiny-1k-224', 'facebook/convnextv2-tiny-22k-224',
                       'facebook/convnextv2-tiny-22k-384',
                       'facebook/data2vec-vision-base-ft1k', 'facebook/deit-base-distilled-patch16-224',
                       'facebook/deit-base-distilled-patch16-384', 'facebook/deit-base-patch16-224',
                       'facebook/deit-base-patch16-384', 'facebook/deit-small-distilled-patch16-224',
                       'facebook/deit-small-patch16-224', 'facebook/deit-tiny-distilled-patch16-224',
                       'facebook/deit-tiny-patch16-224', 'facebook/dinov2-base-imagenet1k-1-layer',
                       'facebook/dinov2-giant-imagenet1k-1-layer', 'facebook/dinov2-large-imagenet1k-1-layer',
                       'facebook/dinov2-small-imagenet1k-1-layer', 'facebook/levit-128S', 'facebook/levit-256',
                       'facebook/regnet-y-040', 'google/bit-50', 'google/efficientnet-b0', 'google/efficientnet-b2',
                       'google/efficientnet-b4', 'google/efficientnet-b7', 'google/mobilenet_v1_0.75_192',
                       'google/mobilenet_v1_1.0_224', 'google/mobilenet_v2_0.35_96', 'google/mobilenet_v2_0.75_160',
                       'google/mobilenet_v2_1.0_224', 'google/mobilenet_v2_1.4_224', 'google/vit-base-patch16-224',
                       'google/vit-base-patch16-224', 'google/vit-base-patch16-224-in21k',
                       'google/vit-base-patch16-384',
                       'google/vit-base-patch32-384', 'google/vit-hybrid-base-bit-384', 'google/vit-large-patch16-224',
                       'google/vit-large-patch16-384', 'google/vit-large-patch32-384',
                       'microsoft/beit-base-patch16-224',
                       'microsoft/beit-base-patch16-224-pt22k', 'microsoft/beit-base-patch16-224-pt22k-ft22k',
                       'microsoft/beit-base-patch16-384', 'microsoft/beit-large-patch16-224',
                       'microsoft/beit-large-patch16-224-pt22k', 'microsoft/beit-large-patch16-224-pt22k-ft22k',
                       'microsoft/beit-large-patch16-512', 'microsoft/cvt-13', 'microsoft/cvt-21-384-22k',
                       'microsoft/dit-base-finetuned-rvlcdip', 'microsoft/focalnet-tiny', 'microsoft/resnet-101',
                       'microsoft/resnet-152', 'microsoft/resnet-18', 'microsoft/resnet-34', 'microsoft/resnet-50',
                       'microsoft/resnet-50', 'microsoft/swin-base-patch4-window12-384',
                       'microsoft/swin-base-patch4-window12-384-in22k', 'microsoft/swin-base-patch4-window7-224',
                       'microsoft/swin-base-patch4-window7-224-in22k', 'microsoft/swin-large-patch4-window12-384-in22k',
                       'microsoft/swin-large-patch4-window7-224', 'microsoft/swin-large-patch4-window7-224-in22k',
                       'microsoft/swin-small-patch4-window7-224', 'microsoft/swin-tiny-patch4-window7-224',
                       'microsoft/swinv2-base-patch4-window12-192-22k', 'microsoft/swinv2-base-patch4-window8-256',
                       'microsoft/swinv2-large-patch4-window12-192-22k',
                       'microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft',
                       'microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft',
                       'microsoft/swinv2-small-patch4-window8-256',
                       'microsoft/swinv2-tiny-patch4-window16-256', 'microsoft/swinv2-tiny-patch4-window8-256',
                       'nateraw/vit-age-classifier', 'nateraw/vit-base-patch16-224-cifar10', 'nvidia/mit-b0',
                       'nvidia/mit-b1',
                       'nvidia/mit-b2', 'nvidia/mit-b3', 'nvidia/mit-b4', 'nvidia/mit-b5',
                       'openai/clip-vit-large-patch14',
                       'sail/poolformer_s12', 'shehan97/mobilevitv2-1.0-imagenet1k-256', 'shi-labs/dinat-mini-in1k-224',
                       'shi-labs/nat-mini-in1k-224', 'vesteinn/vit-mae-cub']
