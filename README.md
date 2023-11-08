# PTM-Recommendation

Using datasets from:

- [PTMTorrent: A Dataset for Mining Open-source Pre-trained Model Packages](https://arxiv.org/abs/2303.08934)

References:

```latex
@inproceedings{you_logme:_2021,
	title = {LogME: Practical Assessment of Pre-trained Models for Transfer Learning},
	booktitle = {ICML},
	author = {You, Kaichao and Liu, Yong and Wang, Jianmin and Long, Mingsheng},
	year = {2021}
}

@article{you_ranking_2022,
	title = {Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs},
	journal = {JMLR},
	author = {You, Kaichao and Liu, Yong and Zhang, Ziyang and Wang, Jianmin and Jordan, Michael I. and Long, Mingsheng},
	year = {2022}
}
```

## Models in use

```python
models = ['apple/mobilevit-small',
          'facebook/convnextv2-tiny-1k-224',
          'facebook/convnextv2-tiny-22k-384',
          'google/mobilenet_v1_0.75_192',
          'google/mobilenet_v2_1.0_224',
          'google/vit-base-patch16-224',
          'google/vit-base-patch16-384',
          'google/vit-large-patch32-384',
          'microsoft/beit-base-patch16-224',
          'microsoft/beit-base-patch16-224-pt22k-ft22k',
          'microsoft/dit-base-finetuned-rvlcdip',
          'microsoft/resnet-18',
          'microsoft/resnet-50',
          'microsoft/swin-base-patch4-window7-224-in22k',
          'microsoft/swin-tiny-patch4-window7-224',
          'nateraw/vit-age-classifier',
          'nvidia/mit-b0',
          'nvidia/mit-b2']
```

## Datasets in use

```python
datasets = [
    {
        'name': 'cifar10',
        'image_key': 'img',
        'label_key': 'label'
    },
    {
        'name': 'cifar100',
        'image_key': 'img',
        'label_key': 'fine_label'
    },
    {
        'name': 'beans',
        'image_key': 'image',
        'label_key': 'labels'
    },
    {
        'name': 'Matthijs/snacks',
        'image_key': 'image',
        'label_key': 'label'
    },
    {
        'name': 'sasha/dog-food',
        'image_key': 'image',
        'label_key': 'label'
    },
]
```