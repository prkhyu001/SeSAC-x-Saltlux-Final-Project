import googletrans
translator = googletrans.Translator()


def trans_to_eng(str):
    return translator.translate(str, dest='en').text



# pip install googletrans==3.1.0a0