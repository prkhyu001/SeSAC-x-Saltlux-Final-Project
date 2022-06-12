import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration


model = BartForConditionalGeneration.from_pretrained('./kobart_chatbot_model')

tokenizer = get_kobart_tokenizer()

def chat(text):
        input_ids =  [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
        res_ids = model.generate(torch.tensor([input_ids]),
                                            max_length=128,
                                            num_beams=5,
                                            eos_token_id=tokenizer.eos_token_id,
                                            bad_words_ids=[[tokenizer.unk_token_id]])        
        a = tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', '')


# print(chat("아 오늘 친구랑 밥 같이 먹다가 식판 엎었어."))