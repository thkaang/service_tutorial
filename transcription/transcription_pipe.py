import io
import time
import librosa
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import OrderedDict, defaultdict
from pyannote.audio import Pipeline
from datetime import timedelta
import core.separate_fast as separate_fast
from core.uvr.separate import _audio_pre_
from core.silero_vad_module import SileroVAD
from core.whisper_api import Whisper
from core.amphion_utils import *
from core.custom_utils import *


class TranscriptionPipe:
    def __init__(self, cfg, device_name='cuda', whisper_model_type='turbo'):
        self.cfg = cfg
        self.device_name = device_name
        self.target_sr = 16000
        self.separate_predictor = None
        self.dia_pipeline = None
        self.tr_pipe_type = cfg['transcription_pipe_type']
        self.whisper_model_path = cfg['whisper_model_path']
        if device_name == 'cuda':
            self.separate_predictor = _audio_pre_(model_path=cfg['separate_gpu']['model_path'], device=device_name, is_half=True)
        else:
            self.separate_predictor = separate_fast.Predictor(args=cfg["separate"]["step1"], device=device_name)

        if self.tr_pipe_type == 'v1':
            self.dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(torch.device(device_name))
        self.silero_vad = SileroVAD(local=True)
        self.whisper = Whisper(type=whisper_model_type, model_path=self.whisper_model_path, device=device_name)

    def source_separation(self, audio):
        target_sr = 44100
        vocals, no_vocals = self.separate_predictor.predict(audio, target_sr)
        audio["sample_rate"] = target_sr
        audio["waveform"] = vocals[:, 0]

        return audio

    def get_audio_chunks(self, audio):
        audio_chunks = []
        resampled_audio = {}
        audio_length = len(audio["waveform"]) / audio["sample_rate"]

        if audio_length > self.cfg["vad"]["preprocess"]["MAX_SEGMENT_LENGTH"]:  # for long audio
            print("Step 1.1: Voice Activity Detection (for long audio)")
            print("Step 1.1.1: resampling for pre vad")
            resampled_audio["waveform"] = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"],
                                                           target_sr=self.target_sr)
            resampled_audio["sample_rate"] = self.target_sr
            print("Step 1.1.2: execute pre vad")
            pre_vad_list = vad_only(self.silero_vad, resampled_audio)
            pre_segment_list = merge_over_min_length(pre_vad_list, self.cfg["vad"]["preprocess"], True)
            for pre_timestamp in pre_segment_list:
                start_frame = int(pre_timestamp["start"] * audio["sample_rate"])
                end_frame = int(pre_timestamp["end"] * audio["sample_rate"])
                start_timestamp = timedelta_to_hms(timedelta(seconds=pre_timestamp["start"]))
                end_timestamp = timedelta_to_hms(timedelta(seconds=pre_timestamp["end"]))
                audio_chunks.append({
                    "waveform": audio["waveform"][start_frame:end_frame],
                    "sample_rate": audio["sample_rate"],
                    "start_time": start_timestamp,
                    "end_time": end_timestamp
                })
        else:
            start_timestamp = timedelta_to_hms(timedelta(seconds=0))
            end_timestamp = timedelta_to_hms(timedelta(seconds=(len(audio["waveform"]) / audio["sample_rate"])))
            audio["start_time"] = start_timestamp
            audio["end_time"] = end_timestamp
            audio_chunks = [audio]

        return audio_chunks

    def run(self, audio_buffer: io.BytesIO, audio_format: str):
        if self.tr_pipe_type == 'v1':
            return self.__run_v1(audio_buffer, audio_format)
        elif self.tr_pipe_type == 'v2':
            return self.__run_v2_wo_spkdia(audio_buffer, audio_format)
        elif self.tr_pipe_type == 'v3':
            return self.__run_v3_batch_wo_spkdia(audio_buffer, audio_format)
        else:
            return f"not supported transcription_pipe_type: {self.tr_pipe_type}"

    def __run_v1(self, audio_buffer: io.BytesIO, audio_format: str):
        start = time.time()
        print("Step 1: standardization")
        audio = standardization(audio_buffer, audio_format)

        print("Step 2: source separation")
        audio = self.source_separation(audio)

        # resample to 16kHz
        audio["waveform"] = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=self.target_sr)
        audio["sample_rate"] = self.target_sr

        print("Step 3: speaker diarization")
        speaker_info_df = speaker_diarization(audio, self.dia_pipeline, device_name=self.device_name)
        print("Step 4: Voice Activity Detection")
        vad_list = self.silero_vad.vad(speaker_info_df, audio)
        segment_list = cut_by_speaker_label(vad_list)
        print("Step 5: Transcription")
        asr_result = asr_whisper(segment_list, audio, self.whisper, restrict_lang_dict=self.cfg["restrict_lang"])
        print("Transcription process finished")
        elapsed_time = time.time() - start
        audio_length = len(audio["waveform"]) / audio["sample_rate"]
        print(f"Total elapsed time: {elapsed_time:.3f} sec")
        print(f"xRT: {elapsed_time / audio_length:.3f} xRT")

        return asr_result

    def __run_v2_wo_spkdia(self, audio_buffer: io.BytesIO, audio_format: str):
        start = time.time()
        asr_results = []
        print("Step 1: standardization")
        audio = standardization(audio_buffer, audio_format)

        audio_chunks = self.get_audio_chunks(audio)

        for idx, audio_chunk in enumerate(tqdm(audio_chunks), start=1):
            # sf.write(f"./audio_data/test_{audio_chunk['name']}_{idx}.wav", audio_chunk["waveform"], samplerate=audio_chunk["sample_rate"])
            print(f"Step 2_({idx}): source separation")
            audio_chunk = self.source_separation(audio_chunk)

            # resample to 16kHz
            print(f"Step 2.1_({idx}): resampling for vad")
            audio_chunk["waveform"] = librosa.resample(audio_chunk["waveform"], orig_sr=audio_chunk["sample_rate"], target_sr=self.target_sr)
            audio_chunk["sample_rate"] = self.target_sr

            print(f"Step 3_({idx}): Voice Activity Detection")
            vad_list = vad_only(self.silero_vad, audio_chunk)
            segment_list = merge_over_min_length(vad_list, self.cfg['vad']['transcription'])
            print(f"Step 4_({idx}): Transcription")
            asr_result = asr_whisper(segment_list, audio_chunk, self.whisper, restrict_lang_dict=self.cfg["restrict_lang"])
            asr_result = f"{audio_chunk['start_time']}~{audio_chunk['end_time']}. {' '.join(asr_result)}\n"
            asr_results.append(asr_result)

        print("Transcription process finished")
        elapsed_time = time.time() - start
        audio_length = len(audio["waveform"]) / audio["sample_rate"]
        print(f"Total elapsed time: {elapsed_time:.3f} sec")
        print(f"xRT: {elapsed_time / audio_length:.3f} xRT")

        return asr_results

    def __run_v3_batch_wo_spkdia(self, audio_buffer: io.BytesIO, audio_format: str):
        start = time.time()
        asr_results = {}
        segment_dict = defaultdict(dict)
        audio_chunk_time_dict = {}
        final_asr_results = []
        print("Step 1: standardization")
        audio = standardization(audio_buffer, audio_format)

        audio_chunks = self.get_audio_chunks(audio)

        for idx, audio_chunk in enumerate(tqdm(audio_chunks), start=1):
            audio_chunk_time_dict[str(idx)] = {'start_time': audio_chunk['start_time'], 'end_time': audio_chunk['end_time']}
            # sf.write(f"./audio_data/test_{audio_chunk['name']}_{idx}.wav", audio_chunk["waveform"], samplerate=audio_chunk["sample_rate"])
            print(f"Step 2_({idx}): source separation")
            audio_chunk = self.source_separation(audio_chunk)

            # resample to 16kHz
            print(f"Step 2.1_({idx}): resampling for vad")
            audio_chunk["waveform"] = librosa.resample(audio_chunk["waveform"], orig_sr=audio_chunk["sample_rate"], target_sr=self.target_sr)
            audio_chunk["sample_rate"] = self.target_sr

            print(f"Step 3_({idx}): Voice Activity Detection")
            vad_list = vad_only(self.silero_vad, audio_chunk)
            segment_list = merge_over_min_length(vad_list, self.cfg['vad']['transcription'])

            for seg_idx, segment in enumerate(segment_list):
                start_frame = int(segment["start"] * 16000)
                end_frame = int(segment["end"] * 16000)

                segment_audio = audio_chunk["waveform"][start_frame:end_frame]
                segment_audio = self.whisper.pad_or_trim(segment_audio)
                mel = self.whisper.log_mel_spectrogram(segment_audio)
                language_code = self.whisper.detect_language(mel)

                restrict_lang_dict = self.cfg["restrict_lang_v3"]

                if restrict_lang_dict:
                    if language_code not in restrict_lang_dict.values():
                        print(f"wrong lang: {language_code}")
                        language_code = restrict_lang_dict["whisper_restrict"]
                        print(f"changed lang: {language_code}")

                segment_dict[language_code][f"{idx}_{seg_idx}"] = [mel, segment["end"] - segment["start"]]

        print(f"Step 4: Transcription")
        for language in segment_dict.keys():
            sorted_mel_segment_dict = OrderedDict(sorted(segment_dict[language].items(), key=lambda x: x[1][1]))
            asr_result = asr_whisper_batch(sorted_mel_segment_dict, self.whisper,
                                           batch_size=self.cfg["whisper_batch_size"], language=language)
            asr_results.update(asr_result)

        ordered_asr_results = OrderedDict(sorted(asr_results.items(),
                                                 key=lambda x: (int(x[0].split("_")[0]), int(x[0].split("_")[1]))))
        ordered_chunk_idx_asr_results = OrderedDict()
        for idx_info, script in ordered_asr_results.items():
            audio_chunk_idx, seg_idx = idx_info.split("_", 1)
            if audio_chunk_idx not in ordered_chunk_idx_asr_results:
                ordered_chunk_idx_asr_results[audio_chunk_idx] = script
            else:
                ordered_chunk_idx_asr_results[audio_chunk_idx] += " " + script

        for chunk_idx, script in ordered_chunk_idx_asr_results.items():
            timestamp_dict = audio_chunk_time_dict[chunk_idx]
            final_asr_result = f"{timestamp_dict['start_time']}~{timestamp_dict['end_time']}\t{script}"
            final_asr_results.append(final_asr_result)

        print("Transcription process finished")
        elapsed_time = time.time() - start
        audio_length = len(audio["waveform"]) / audio["sample_rate"]
        print(f"Total elapsed time: {elapsed_time:.3f} sec")
        print(f"xRT: {elapsed_time / audio_length:.3f} xRT")

        return final_asr_results
