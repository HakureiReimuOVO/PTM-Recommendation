from config import *
from model_loader import get_model
from ptflops import get_model_complexity_info

# def get_model(model_name):
#     model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}", ignore_mismatched_sizes=True)
#     return model

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == '__main__':
    input_size = (3, 224, 224)
    for model_config in model_configs:
        model = get_model(model_config)
        params, trainable_params = get_model_params(model)
        flops, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        print(model_config)
        print(f'Total params: {params}, Flops: {flops}')
