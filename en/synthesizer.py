import io

import numpy as np
import tensorflow as tf
from en.hparams import hparams
from en.models import create_model
from en.text import text_to_sequence
from en.util import audio


class Synthesizer:
    isModelLoaded = False

    def load(self, type='EN', model_name='tacotron'):
        checkpoint_path = None
        if type == 'EN':
            checkpoint_path = 'logs-tacotron/model.ckpt-en-60000'
        elif type == 'CN':
            checkpoint_path = 'logs-tacotron/model.ckpt-cn-70000'

        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs, input_lengths)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

        print('Loading checkpoint: %s' % checkpoint_path)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)
        if type == 'EN':
            self.session_EN = session
        elif type == 'CN':
            self.session_CN = session

    def synthesize(self, text, type):
        if type == 'EN':
            cleaner_names = [x.strip() for x in hparams.cleaners[0].split(',')]
        else:
            cleaner_names = [x.strip() for x in hparams.cleaners[1].split(',')]
        seq = text_to_sequence(text, cleaner_names, type)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
        }
        wav = None
        if type == 'EN' and self.session_EN:
            wav = self.session_EN.run(self.wav_output, feed_dict=feed_dict)
        elif type == 'CN' and self.session_CN:
            wav = self.session_CN.run(self.wav_output, feed_dict=feed_dict)
        if wav is not None:
            wav = audio.inv_preemphasis(wav)
            out = io.BytesIO()
            audio.save_wav(wav, out)
            return out.getvalue()
        else:
            return None
