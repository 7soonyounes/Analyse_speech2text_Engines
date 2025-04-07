import time
import torch
import psutil
import os
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import wer

num_samples = 50  
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


datasets = {
    # "librispeech_en": load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True),
    "fleurs_fr": load_dataset("google/fleurs", "fr_fr", split="test", trust_remote_code=True),
}

whisper_models = {
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
}

# Fonction pour mesurer l'utilisation des ressources
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) 

def test_model(model_name, model_id, dataset, num_samples):
    print(f"Test du modèle {model_name}...")
    start_mem = get_process_memory()
    

    if "whisper" in model_name:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
    
    
    model_size = get_process_memory() - start_mem
    
    wer_scores = []
    execution_times = []
    
    for i in range(min(num_samples, len(dataset))):
        # reference = dataset[i]["text"]
        # audio = dataset[i]["audio"]["array"]
        reference = dataset[i]["transcription"]  # pour Fleurs dataset
        audio = dataset[i]["audio"]["array"]
        
        start_time = time.time()
        result = pipe(audio)
        end_time = time.time()
        
        transcription = result["text"]
        error_rate = wer(reference, transcription)
        execution_time = end_time - start_time
        
        wer_scores.append(error_rate)
        execution_times.append(execution_time)
        
        if i % 10 == 0:
            print(f"  Échantillon {i+1}/{min(num_samples, len(dataset))}")
    
    avg_wer = np.mean(wer_scores) * 100  
    avg_time = np.mean(execution_times)
    
    return {
        "model": model_name,
        "wer": avg_wer,
        "execution_time": avg_time,
        "model_size_mb": model_size
    }

def evaluate_models():
    results = []
    
    for dataset_name, dataset in datasets.items():
        print(f"Évaluation sur le dataset {dataset_name}")
        
        for model_name, model_id in whisper_models.items():
            try:
                result = test_model(model_name, model_id, dataset, num_samples)
                result["dataset"] = dataset_name
                results.append(result)
            except Exception as e:
                print(f"Erreur avec le modèle {model_name}: {e}")

    print("\n" + "_"*77)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("_"*77)
    
    for dataset_name in datasets.keys():
        print(f"\nRésultats pour {dataset_name}:")
        print(f"{'Modèle':<25} {'WER (%)':<10} {'Temps (s)':<12} {'Taille (MB)':<12}")
        print("-"*60)
        
        dataset_results = [r for r in results if r["dataset"] == dataset_name]
        dataset_results.sort(key=lambda x: x["wer"]) 
        
        for r in dataset_results:
            print(f"{r['model']:<25} {r['wer']:<10.2f} {r['execution_time']:<12.2f} {r['model_size_mb']:<12.1f}")
    
    return results

if __name__ == "__main__":
    results = evaluate_models()