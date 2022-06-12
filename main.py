
import kobart_chatbot as kochat
import text_rw as text_record
import summary_model_infer as smi
import hajoon_tts as htts # 하준 tts(한글)
import translator as trans # 번역
import gTTS as gtts # 구글 tts (영어)
import multi_sentiment_cls as msc # 감정분석


# 순차적으로 실행

STT_text = "오늘 안으로 프로젝트 끝났으면 좋겠다."

result_text = kochat.chat(STT_text) # 챗봇 대답
print(result_text)

trans_result = trans.trans_to_eng(result_text) # 영어로 번역
print("번역:",trans_result)

htts.generate_tts(result_text) # 하준의 목소리 wav

gtts.speech_gtts(trans_result) # gtts 영어 목소리 wav


text_record.write_text_user(STT_text.strip()) # .txt저장 > 아이 내용만 저장 (한국어만)

text_record.write_text_user_chatbot(STT_text.strip(), trans.trans_to_eng(STT_text).strip(),result_text.strip() ,trans.trans_to_eng(result_text).strip()) # .txt저장 > 아이 내용 Ko/Eng & 챗봇 Ko/Eng


# 요약을 원한다면 
text = text_record.read_text_user().strip() # 아이(+챗봇?)대화 읽어오기
# print(text)
summary_result = smi.start_infer("\n"+text+"\n") # 대화 한줄 요약

print(summary_result)

sentiment_result = msc.predict(summary_result)
print(sentiment_result)

print(text_record.read_text_user_chatbot().strip())

    # 0524 requirement 설치
    # g2pk 폴더 > python setup.py install
    # TTS 폴더 > python setup.py insq