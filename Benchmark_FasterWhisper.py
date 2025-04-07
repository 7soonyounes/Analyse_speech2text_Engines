import os
import time
import torch
import psutil
from datasets import load_dataset
from jiwer import wer
from faster_whisper import WhisperModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8"
num_samples = 50 

# Dataset
dataset = load_dataset("google/fleurs", "fr_fr", split="test", trust_remote_code=True)

# Modèles 
faster_whisper_models = {
    "tiny": "tiny",
    "medium": "medium",
    "large-v3": "large-v3"
}

# Fonction mémoire
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # en MB

# Fonction de test
def test_faster_whisper(model_name, model_id, dataset, num_samples):
    print(f"\n=== Test du modèle FasterWhisper : {model_name} ===")
    start_mem = get_process_memory()

    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    mem_after_load = get_process_memory()
    model_size = mem_after_load - start_mem
    print(f"[MEMOIRE] Utilisation mémoire après chargement : {model_size:.2f} MB")

    wers = []
    exec_times = []

    for i, sample in enumerate(dataset.select(range(num_samples))):
        audio = sample["audio"]["array"]
        reference = sample["transcription"].lower().strip()

        start_time = time.time()
        segments, _ = model.transcribe(audio, beam_size=5)
        duration = time.time() - start_time

        transcription = " ".join([seg.text for seg in segments]).lower().strip()
        error = wer(reference, transcription)

        wers.append(error)
        exec_times.append(duration)

        print(f"[{i+1}/{num_samples}] WER: {error:.3f} | Durée: {duration:.2f}s")
        print(f"Réf  : {reference}")
        print(f"Trans: {transcription}")
        print("-" * 50)

    avg_wer = sum(wers) / len(wers)
    avg_time = sum(exec_times) / len(exec_times)

    print(f"\n--- Résultats pour {model_name} ---")
    print(f"--> Moyenne WER: {avg_wer:.3f}")
    print(f"--> Temps moyen de transcription: {avg_time:.2f}s")
    print(f"--> Mémoire utilisée (chargement modèle): {model_size:.2f} MB\n")
    print("=" * 70)

for name, model_id in faster_whisper_models.items():
    test_faster_whisper(name, model_id, dataset, num_samples)
