import glob
import shutil
from paddleocr import PaddleOCR
import wave
import json
from moviepy.editor import *
from vosk import Model, KaldiRecognizer, SetLogLevel
from openai import OpenAI


def ocr():
    # 初始化PaddleOCR，使用简体中文和英文模型
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    images_path = glob.glob(os.path.join('./image/' + '*.png'))
    print(images_path)
    result = []
    for image_path in images_path:
        # 进行OCR识别
        result.extend(ocr.ocr(image_path, cls=True))
        print(image_path)

    res_text = ''

    for line in result:
        for element in line:
            # 识别到的文本
            text = element[1][0]

            res_text += text

    return res_text


def read_video(video_name, segment_duration=20):
    video = VideoFileClip(video_name)
    audio_clip = video.audio

    for t in range(0, int(video.duration), segment_duration):
        # 定义图片的文件名，格式为 时间.png
        imgpath = os.path.join('./image', "{}.png".format(t))
        # 使用 save_frame 方法，传入图片文件名和时间参数
        video.save_frame(imgpath, t)
        print(f'成功截取图片："{t}.png"')

    # 计算音频的总长度
    audio_duration = audio_clip.duration

    # 计算可以分割的片段数量
    num_segments = int(audio_duration // segment_duration)

    # 遍历每个片段并保存
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        # 创建子剪辑
        segment_clip = audio_clip.subclip(start_time, end_time)
        # 保存子剪辑为单独的音频文件
        # 使用i来为每个文件命名，确保文件名唯一
        segment_clip.write_audiofile(f'./audio/audio_segment_{i + 1}.wav', ffmpeg_params=["-ac", "1"])

    # 如果音频的总长度不是20秒的整数倍，处理剩余的部分
    remaining_time = audio_duration % segment_duration
    if remaining_time > 0:
        start_time = num_segments * segment_duration
        remaining_clip = audio_clip.subclip(start_time, audio_duration)
        remaining_clip.write_audiofile(f'./audio/audio_segment_{num_segments + 1}.wav', ffmpeg_params=["-ac", "1"])


def audio2txt():
    # You can set log level to -1 to disable debug messages
    SetLogLevel(-1)
    model = Model("model-small")
    str_ret = ""
    i = 0
    while True:
        try:
            wf = wave.open(f'./audio/audio_segment_{i + 1}.wav', "rb")
        except:
            break
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            break

        # model = Model(lang="en-us")
        # You can also init model by name or with a folder path
        # model = Model(model_name="vosk-model-en-us-0.21")
        # 设置模型所在路径，刚刚4.1中解压出来的路径   《《《《
        # model = Model("model")

        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        # rec.SetPartialWords(True)   # 注释这行   《《《《
        print(f'正在转换./audio/audio_segment_{i + 1}.wav')

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = rec.Result()
                # print(result)

                result = json.loads(result)
                if 'text' in result:
                    str_ret += result['text'] + ' '
            # else:
            #     print(rec.PartialResult())

        # print(rec.FinalResult())
        result = json.loads(rec.FinalResult())
        if 'text' in result:
            str_ret += result['text']
        i += 1

    return str_ret


def llm_format(text, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": '以下输入的内容是由语音转文字或图片转文字生成的，请：\n\
             1.纠错并转换为正确的格式\n\
             2.去除个人信息，如视频创作者名称等\n\
             3.去除无关内容，如求关注收藏等\n\
             4.不要说任何多余的话，如："以下是..."或"请注意..."\n\
             5.尽可能保留运行结果和效果'},
            {"role": "user", "content": text},
        ]
    )
    return response.choices[0].message.content


def save_text(path, text):
    with open(path, 'w', encoding='utf-8') as file:  # 使用'w'模式，如果文件存在则覆盖
        file.write(text)


def llm_to_note(audio, image, client):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system",
             "content": '参考以下输入的内容，请：\n\
                 1.转换为MarkDown格式\n\
                 2.去除代码中关于个人信息的注释\n\
                 3.转换为一份详细的教程，包含背景，每一步的说明，每一步的代码（如有），每一步的运行结果（如有），和完整代码\n\
                 4.禁止说任何多余的话，如："以下是..."或"请注意..."\n\
                 5.分步思考，每一步都要详细解释并附代码（如有）'

             },
            {"role": "user", "content": audio + image},
        ]
    )
    return response.choices[0].message.content


def main():
    video_name = 'video3.mp4' # 视频路径
    segment_duration = 20 # 片段时长 20s
    client = OpenAI(api_key="Your API key", base_url="https://api.deepseek.com") # 初始化大模型API

    # 删除原有缓存目录
    try:
        shutil.rmtree('./audio')
        shutil.rmtree('./image')
    except:
        pass

    # 创建新缓存目录
    try:
        os.makedirs('audio')
        os.makedirs('image')
    except:
        pass

    read_video(video_name, segment_duration)
    ocr_res = ocr()
    save_text('original_ocr_res.txt', ocr_res) # ocr结果
    fix_ocr_res = llm_format(ocr_res, client)
    save_text('ocr_res.txt', fix_ocr_res) # 大模型处理后的ocr结果
    audio_res = audio2txt()
    audio_res = audio_res.replace(' ', '')
    save_text('original_audio_res.txt', audio_res) # 语音转文字结果
    fix_audio_res = llm_format(audio_res, client)
    save_text('audio_res.txt', fix_audio_res) # 大模型处理后的语音转文字结果
    note = llm_to_note(fix_audio_res, fix_ocr_res, client)
    save_text(video_name.replace('.mp4', '_') + 'note.md', note) # 最终教程md


if __name__ == '__main__':
    main()
