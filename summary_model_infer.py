import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
tokenizer = get_kobart_tokenizer()
def start_infer(input_text):
    text = input_text
    text = text.replace('\n', ' ')
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output)
    return output

# exam_text = '''
# 유튜브가 너무 재밌어서 고민이야. 공부보다 더 재밌어.
# 유튜브를 보면서 스트레스를 풀 수 있어.
# 엄마는 내가 공부를 안하고 유튜브만 볼까봐 걱정하는 거 같아.
# 내 학교 점수가 떨어질까봐… 나는 유튜브 조금만 보고, 공부 열심히 할 수 있는데.
# # '''
# start_infer(exam_text)
# print(exam_text)
# print(len(exam_text.strip().split('\n')))
# start_infer("\n" + "\n".join(exam_text.strip().split('\n')) + '\n')
# start_infer("\n" + "\n".join(exam_text.strip().split('\n')[22:44]) + '\n')
# start_infer("\n" + "\n".join(exam_text.strip().split('\n')[44:]) + '\n')
# print("\n".join(exam_text.strip().split('\n')[:]))
#pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
# python get_model_binary.py --hparams KoBART-summarization\logs\train_model_extract\3-epoch-hparams.yaml --model_binary KoBART-summarization\logs\train_model_extract\epoch=03-val_loss=2.019.ckpt