import requests
import re
from bs4 import BeautifulSoup


def crawl_models(page_list=None, sort=False):
    if page_list is None:
        page_list = [0]
    models = []
    for page in page_list:
        params = {
            'pipeline_tag': 'image-classification',
            'library': 'transformers',
            'sort': 'downloads',
            'p': str(page)
        }

        url = 'https://huggingface.co/models'
        response = requests.get(url, params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        model_sections = soup.find_all('article', class_='overview-card-wrapper')

        for result in model_sections:
            href = result.find_next('a').get('href')
            model_url = 'https://huggingface.co' + href
            model = get_model_info(model_url)
            for model_name in model:
                models.append(model_name)

    if sort:
        models = sorted(models)
    return models


def get_model_info(model_url):
    response = requests.get(model_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    codes = soup.find_all('code', class_='language-python')

    pattern = r"from_pretrained\((['\"])([^'\"]*?)\1\)"

    result = set()
    for code in codes:
        matches = re.findall(pattern, code.text)
        for match in matches:
            result.add(match[1])

    return result


if __name__ == '__main__':
    res = crawl_models(page_list=list(range(5)), sort=True)
    print(res)
    # _ = ['apple/mobilevit-small',
    #      'facebook/convnextv2-tiny-1k-224',
    #      'facebook/convnextv2-tiny-22k-384',
    #      'google/mobilenet_v1_0.75_192',
    #      'google/mobilenet_v2_1.0_224',
    #      'google/vit-base-patch16-224',
    #      'google/vit-base-patch16-384',
    #      'google/vit-large-patch32-384',
    #      'microsoft/beit-base-patch16-224',
    #      'microsoft/beit-base-patch16-224-pt22k-ft22k',
    #      'microsoft/dit-base-finetuned-rvlcdip',
    #      'microsoft/resnet-18',
    #      'microsoft/resnet-50',
    #      'microsoft/swin-base-patch4-window7-224-in22k',
    #      'microsoft/swin-tiny-patch4-window7-224',
    #      'nateraw/vit-age-classifier',
    #      'nvidia/mit-b0',
    #      'nvidia/mit-b2']
    pass