import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

a = {'microsoft/resnet-18': {'microsoft/resnet-50': {'weight': 0.65895694},
                             'facebook/convnextv2-tiny-1k-224': {'weight': -0.032781694},
                             'microsoft/swin-tiny-patch4-window7-224': {'weight': 0.0382542},
                             'google/mobilenet_v2_1.0_224': {'weight': 0.7786167},
                             'google/vit-base-patch16-224': {'weight': -0.1186306},
                             'google/efficientnet-b0': {'weight': 0.59551287},
                             'microsoft/beit-base-patch16-224': {'weight': 0.035677984}},
     'microsoft/resnet-50': {'microsoft/resnet-18': {'weight': 0.65895694},
                             'facebook/convnextv2-tiny-1k-224': {'weight': 0.011862616},
                             'microsoft/swin-tiny-patch4-window7-224': {'weight': 0.026691815},
                             'google/mobilenet_v2_1.0_224': {'weight': 0.72577804},
                             'google/vit-base-patch16-224': {'weight': -0.07629919},
                             'google/efficientnet-b0': {'weight': 0.54587835},
                             'microsoft/beit-base-patch16-224': {'weight': 0.021915305}},
     'facebook/convnextv2-tiny-1k-224': {'microsoft/resnet-18': {'weight': -0.032781694},
                                         'microsoft/resnet-50': {'weight': 0.011862616},
                                         'microsoft/swin-tiny-patch4-window7-224': {'weight': 0.0124210715},
                                         'google/mobilenet_v2_1.0_224': {'weight': -0.017632343},
                                         'google/vit-base-patch16-224': {'weight': 0.03978189},
                                         'google/efficientnet-b0': {'weight': -0.033173483},
                                         'microsoft/beit-base-patch16-224': {'weight': 0.04371451}},
     'microsoft/swin-tiny-patch4-window7-224': {'microsoft/resnet-18': {'weight': 0.0382542},
                                                'microsoft/resnet-50': {'weight': 0.026691815},
                                                'facebook/convnextv2-tiny-1k-224': {'weight': 0.0124210715},
                                                'google/mobilenet_v2_1.0_224': {'weight': 0.001283966},
                                                'google/vit-base-patch16-224': {'weight': -0.005237274},
                                                'google/efficientnet-b0': {'weight': -0.016001867},
                                                'microsoft/beit-base-patch16-224': {'weight': -0.022022374}},
     'google/mobilenet_v2_1.0_224': {'microsoft/resnet-18': {'weight': 0.7786167},
                                     'microsoft/resnet-50': {'weight': 0.72577804},
                                     'facebook/convnextv2-tiny-1k-224': {'weight': -0.017632343},
                                     'microsoft/swin-tiny-patch4-window7-224': {'weight': 0.001283966},
                                     'google/vit-base-patch16-224': {'weight': -0.1152638},
                                     'google/efficientnet-b0': {'weight': 0.6408814},
                                     'microsoft/beit-base-patch16-224': {'weight': 0.04460957}},
     'google/vit-base-patch16-224': {'microsoft/resnet-18': {'weight': -0.1186306},
                                     'microsoft/resnet-50': {'weight': -0.07629919},
                                     'facebook/convnextv2-tiny-1k-224': {'weight': 0.03978189},
                                     'microsoft/swin-tiny-patch4-window7-224': {'weight': -0.005237274},
                                     'google/mobilenet_v2_1.0_224': {'weight': -0.1152638},
                                     'google/efficientnet-b0': {'weight': -0.11500004},
                                     'microsoft/beit-base-patch16-224': {'weight': -0.00546955}},
     'google/efficientnet-b0': {'microsoft/resnet-18': {'weight': 0.59551287},
                                'microsoft/resnet-50': {'weight': 0.54587835},
                                'facebook/convnextv2-tiny-1k-224': {'weight': -0.033173483},
                                'microsoft/swin-tiny-patch4-window7-224': {'weight': -0.016001867},
                                'google/mobilenet_v2_1.0_224': {'weight': 0.6408814},
                                'google/vit-base-patch16-224': {'weight': -0.11500004},
                                'microsoft/beit-base-patch16-224': {'weight': 0.045776136}},
     'microsoft/beit-base-patch16-224': {'microsoft/resnet-18': {'weight': 0.035677984},
                                         'microsoft/resnet-50': {'weight': 0.021915305},
                                         'facebook/convnextv2-tiny-1k-224': {'weight': 0.04371451},
                                         'microsoft/swin-tiny-patch4-window7-224': {'weight': -0.022022374},
                                         'google/mobilenet_v2_1.0_224': {'weight': 0.04460957},
                                         'google/vit-base-patch16-224': {'weight': -0.00546955},
                                         'google/efficientnet-b0': {'weight': 0.045776136}}}


# 转换为相似度矩阵
similarity_matrix = pd.DataFrame(index=a.keys(), columns=a.keys())
for key, values in a.items():
    for sub_key, sub_values in values.items():
        similarity_matrix.loc[key, sub_key] = sub_values['weight']
        similarity_matrix.loc[sub_key, key] = sub_values['weight']  # 确保矩阵对称

# 将缺失值填充为0
similarity_matrix = similarity_matrix.fillna(0)

# 绘制高分辨率热图
plt.figure(figsize=(20, 16), dpi=300)  # 设置画布尺寸和分辨率
sns.heatmap(similarity_matrix.astype(float), cmap='viridis', square=True, cbar_kws={"shrink": .82})
plt.xticks(rotation=90, fontsize=18)  # 调整x轴字体大小
plt.yticks(rotation=0, fontsize=18)  # 调整y轴字体大小
plt.title('Similarity Matrix', fontsize=24)  # 添加标题并调整字体大小
plt.xlabel('Models', fontsize=20)  # 设置x轴标签并调整字体大小
plt.ylabel('Models', fontsize=20)  # 设置y轴标签并调整字体大小
plt.savefig('heatmap_high_res.png', dpi=300, bbox_inches='tight')  # 保存高分辨率图片
plt.show()