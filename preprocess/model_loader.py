from transformers import (AutoModel,
                          AutoImageProcessor,
                          AutoModelForImageClassification)


def get_hf_model_and_processor(model_name, print_info=False):
    model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}")
    processor = AutoImageProcessor.from_pretrained(f"models/{model_name}")
    if print_info:
        print('Model preview:')
        print(model)
        print('Processor preview:')
        print(processor)
    if print_info:
        print('Model and processor loaded successfully.')
    return model, processor
