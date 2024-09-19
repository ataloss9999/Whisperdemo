#This is a simple demo of the Open-AI's Whisper transcription functionality.

#Requirements :
#Python3+
#FFMPEG (sudo apt update && sudo apt install ffmpeg, manually download at https://www.ffmpeg.org/download.html for windows*)
#openai-whisper (pip install -U openai-whisper)
#torch-cuda (Go to https://pytorch.org/get-started/locally/ for customized pip install command)
#Cuda 12.4.0 (Go to https://developer.nvidia.com/cuda-toolkit-archive to download "CUDA Toolkit 12.4.0 (March 2024)")

from os import system, name
import torch
import whisper
import whisper.utils

#Note: recording_loaded mp3 MUST be in the same folder as this program
recording_loaded = "nameof.mp3" #Put the name of the recording here
torchtest = torch.rand(10, 10)
cudatest = torch.cuda.get_rng_state()
model = 0

#Defining the clear function
def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

print(f"This is a demo of OpenAI's Whisper")

#Test for CUDA
if torch.cuda.is_available():
    model = whisper.load_model("medium.en", device="cuda")
    clear()
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA Version: {torch.version.cuda}")
    print(cudatest)
    print("Now running model...")
else:
    model = whisper.load_model("small.en")
    clear()
    print("CUDA is not available! NOTE: requires CUDA 12.4. Using CPU, expect slowdown.")
    print(torchtest)
    print("Now running model...")

#Defining result as what the model transcribes from song_loaded
result = model.transcribe(recording_loaded)
clear()
print(result["text"])