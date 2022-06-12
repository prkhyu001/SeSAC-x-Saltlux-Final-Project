from gtts import gTTS
from pydub import AudioSegment


def speech_gtts(en_text):
    eng_ex = gTTS(en_text) # 영어 부분
    eng_ex.save('mix-en.wav')


    w1 = AudioSegment.from_mp3('mix-en.wav')
    w2 = AudioSegment.from_mp3('mix.wav')
    combined_w = w2+w1
    combined_w.export('combied_sound.wav',format='wav')

#pip install pydub
#sudo apt update
#sudo apt install ffmpeg 헤야됨.!!!!! 꼭




#pip install gTTS

#FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'