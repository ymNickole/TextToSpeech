import os

from flask import Flask, request, json, make_response, send_from_directory
from pypinyin import pinyin, Style

from en.synthesizer import Synthesizer as Synthesizer_en
from cn.synthesizer import Synthesizer as Synthesizer_cn
from en.util import audio as audio_en
from cn.util import audio as audio_cn

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
synthesizer_en = Synthesizer_en()
# synthesizer_cn = Synthesizer_cn()


@app.route('/connect', methods=['GET'])
def connect():
    return 'Successfully connected!'


@app.route('/loadmodel', methods=['GET'])
def load_model():
    if not synthesizer_en.isModelLoaded:
        synthesizer_en.load('EN')
        synthesizer_en.isModelLoaded = True
    # if not synthesizer_cn.isModelLoaded:
    #     synthesizer_cn.load('CN')
    #     synthesizer_cn.isModelLoaded = True
    return 'TTS engine is loaded successfully!'

@app.route('/text2speech', methods=['POST'])
def text2speech():
    data = json.loads(request.get_data())
    print(data)
    text_id = data["id"]
    type = data["type"]
    type = 'EN'
    standstill_time = data["standstill_time"]
    text_contents = data["text_list"]
    wav_files = []

    for i, text in enumerate(text_contents):
        temp_filename = '%s_%d.wav' % (text_id, i)
        print('Synthesizing: %s' % temp_filename)
        with open(temp_filename, 'wb') as f:
            if type == 'CN':
                pinyin_list = pinyin(text, style=Style.TONE3)
                text = ''
                for py in pinyin_list:
                    if not py[0][-1].isdigit():
                        py[0] = py[0] + '5'
                    text = text + py[0] + ' '
                text.rstrip()
                print(text)
            #     wav_out = synthesizer_cn.synthesize(text, type)
            # else:
            wav_out = synthesizer_en.synthesize(text, type)
            f.write(wav_out)
            wav_files.append(f)
    if len(wav_files) > 1:
        # output_filename = "%s_%s_result.wav" % (text_id, time.strftime("%Y%m%d%H%M%S", time.localtime()))
        output_filename = "%s.wav" % text_id
        # if type == 'CN':
        #     audio_cn.merge_wavs_with_standstill(wav_files, standstill_time, output_filename)
        # else:
        audio_en.merge_wavs_with_standstill(wav_files, standstill_time, output_filename)

    else:
        output_filename = wav_files[0].name
    try:
        # response = make_response(output_file)
        # response.headers['Content-Type'] = 'multipart/form-data'
        # response.headers['Content-Disposition'] = 'attachment; filename={}'.format(
        #     output_filename.encode().decode('latin-1'))
        directory = os.getcwd()  # 获取当前目录
        response = make_response(send_from_directory(directory, output_filename, as_attachment=True))
        response.headers['Content-Disposition'] = 'attachment; filename={}'.format(
            output_filename.encode().decode('latin-1'))
        return response
    except Exception as err:
        print('download wav file error: {}'.format(str(err)))
        return 'Failed to download wav file !'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)




