from transformers import AutoImageProcessor, AutoModelForImageClassification


def get_model_and_processor(model_name, num_labels=2, print_info=False):
    model = AutoModelForImageClassification.from_pretrained(f"models/{model_name}", num_labels=num_labels,
                                                            ignore_mismatched_sizes=True)
    processor = AutoImageProcessor.from_pretrained(f"models/{model_name}")
    if print_info:
        print('Model preview:')
        print(model)
        print('Processor preview:')
        print(processor)
    if print_info:
        print('Model and processor loaded successfully.')
    return model, processor
