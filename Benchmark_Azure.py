import sys
import time
import torch
from jiwer import wer
from datasets import load_dataset
import azure.cognitiveservices.speech as speechsdk
import wave
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

AZURE_KEY = "abc74493841c413b972a35552175aff4"
AZURE_REGION = "swedencentral"
language = "fr-FR"
num_samples = 50

# Dataset
dataset = load_dataset("google/fleurs", "fr_fr", split="test")

speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
speech_config.speech_recognition_language = language


def transcribe_with_azure(audio_array, sample_rate):
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)  

    if len(audio_array.shape) > 1:  
        audio_array = audio_array.mean(axis=1).astype(np.int16)

    push_stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Write the audio array to the push stream
    push_stream.write(audio_array.tobytes())  # Convert numpy array to bytes
    push_stream.close()

    start = time.time()
    result = recognizer.recognize_once()  
    # print("result", result)  
    duration = time.time() - start

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text, duration
    else:
        print(f"Azure Speech SDK failed to recognize speech. Reason: {result.reason}")
        return "", duration



wers = []
durations = []

for i, sample in enumerate(dataset.select(range(num_samples))):
    reference = sample["transcription"]
    audio = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]

    print(f"\n[Sample {i+1}]")

    transcription, duration = transcribe_with_azure(audio, sample_rate)
    error = wer(reference, transcription)

    print(f"Réf  : {reference}")
    print(f"Trans: {transcription}")
    print(f"WER  : {error:.3f} | Durée : {duration:.2f}s")

    wers.append(error)
    durations.append(duration)


print("\n=== Résultats Azure Speech ===")
print(f"--> Moyenne WER: {sum(wers)/len(wers):.3f}")
print(f"--> Temps moyen de transcription: {sum(durations)/len(durations):.2f}s")
