from datetime import datetime
import os

def write_text_user(text):
    now = datetime.today().date()
    
    path = './record_text'
    #폴더 유무 확인
    if os.path.isdir(path): # 이미 폴더 존재
        pass
    else:
        os.mkdir(path)
    
    file_name = str(now) + ".txt"
    # print(file_name)
    with open(f'{path}/{file_name}','a',encoding="UTF-8") as f: # 파일 여부 유무에 따르지 않음
        f.write(text+"\n")

    path = './record_text_user'
    #폴더 유무 확인
    if os.path.isdir(path): # 이미 폴더 존재
        pass
    else:
        os.mkdir(path)
    
    file_name = str(now) + "_user.txt"
    # print(file_name)
    with open(f'{path}/{file_name}','a',encoding="UTF-8") as f: # 파일 여부 유무에 따르지 않음
        f.write(text+"\n")

def write_text_user_chatbot(text,text_eng,chat_text,chat_text_eng):
    now = datetime.today().date()

    path = './record_text_user_chatbot'
    #폴더 유무 확인
    if os.path.isdir(path): # 이미 폴더 존재
        pass
    else:
        os.mkdir(path)
    
    file_name = str(now) + "_user_chatbot.txt"
    # print(file_name)
    with open(f'{path}/{file_name}','a',encoding="UTF-8") as f: # 파일 여부 유무에 따르지 않음
        f.write(text+"\n")
        f.write(text_eng+"\n")
        f.write(chat_text+"\n")
        f.write(chat_text_eng+"\n")


def read_text_user(now_day = 1):
    now = datetime.today().date()
    
    path = './record_text_user'
    #폴더 유무 확인
    if os.path.isdir(path): # 이미 폴더 존재
        pass
    else:
        # print("없음")
        return("데이터가 없습니다.")
    
    file_name = str(now) + "_user.txt"
    # 파일 여부 유무에 따르지 않음
    with open(f'{path}/{file_name}','r',encoding="UTF-8") as f: 
        return(f.read().strip()) 


def read_text_user_chatbot(now_day = 1):
    now = datetime.today().date()
    
    path = './record_text_user_chatbot'
    #폴더 유무 확인
    if os.path.isdir(path): # 이미 폴더 존재
        pass
    else:
        # print("없음")
        return("데이터가 없습니다.")
    
    file_name = str(now) + "_user_chatbot.txt"
    # 파일 여부 유무에 따르지 않음
    with open(f'{path}/{file_name}','r',encoding="UTF-8") as f: 
        return(f.read().strip()) 
