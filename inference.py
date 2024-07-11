import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import shutil
import time
import copy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from config import ex
from model.face_tts import FaceTTS
from model.myface_tts import MyFaceTTS
from data import _datamodules

import numpy as np
from scipy.io.wavfile import write

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils.tts_util import intersperse
import cv2

from tqdm import tqdm
def create_clean_directory(dir_path):
    if os.path.exists(dir_path):
        print(f"Directory {dir_path} already exists. Removing it.")
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"Directory {dir_path} created.")

def save_image(img, name):
    from PIL import Image
    
    array = img
    # (3, 224, 224) -> (224, 224, 3)
    array = array.transpose(1, 2, 0)
    image = Image.fromarray(array)
    image.save(f'/home/jungji/facetts/img/{name}.png')
    
def save_mel(mel_spectrogram,name):
    import matplotlib.pyplot as plt

    mel_spectrogram = mel_spectrogram.cpu().numpy()[0]

    # mel-spectrogram을 이미지로 저장
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel frequency')
    plt.tight_layout()

    # 이미지 파일로 저장
    plt.savefig(f'{name}.png')
    plt.close()
@ex.automain
def main(_config):
    
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    freeze_encoder=True
    print("######## Initializing TTS model")
         
    model = MyFaceTTS(_config,teacher=True,freeze_encoder=freeze_encoder).cuda()
    _config["resume_from"]="/mnt/bear2/users/jungji/facetts_freeze/logs/4/epoch=14-step=2205-last.ckpt"
    model.load_model(_config['resume_from'])
        
    # model = FaceTTS(_config).cuda()
    # _config['resume_from']="facetts_lrs3.pt"
    # model.load_state_dict(torch.load(_config['resume_from'],map_location='cuda:0')['state_dict'])
    
    if isinstance(model, FaceTTS):
        create_clean_directory("original")
    else:
        create_clean_directory("my")
    for n_timesteps in range(1,11):
        if _config['use_custom']:      
            print(f"######## Load {_config['test_faceimg']}")
            # use custom face image to synthesize the speech
            spk = cv2.imread(os.path.join(f"{_config['test_faceimg']}"))
            spk = cv2.resize(spk, (224, 224))
            spk = np.transpose(spk, (2, 0, 1)) #(3,224,224)짜리 int어레이
            # save_image(spk, "test_faceimg")
            spk = torch.FloatTensor(spk).unsqueeze(0).to(model.device) #(1,3,224,224)짜리 int어레이
        else:
            # use LRS3 image 
            print(f"######## Load {_config['dataset']}")
            dm = _datamodules[f"dataset_{_config['dataset']}"](_config)
            dm.set_test_dataset()
            sample = dm.test_dataset[0] # you can adjust the index of test sample
            spk = sample['spk'].to(model.device)

        print(f"######## Load checkpoint from {_config['resume_from']}")
        _config['enc_dropout'] = 0.0
           
        model.eval()
        model.zero_grad()

        print("######## Initializing HiFi-GAN")
        vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().cuda()

        print(f"######## Load text description from {_config['test_txt']}")
        with open(_config['test_txt'], 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]

        cmu = cmudict.CMUDict(_config['cmudict_path'])

        with torch.no_grad():
            
            for i, text in tqdm(enumerate(texts)):
                x = torch.LongTensor(
                    intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
                ).to(model.device)[None]
                
                x_len = torch.LongTensor([x.size(-1)]).to(model.device)
                start_time = time.time()
                y_enc, y_dec, attn = model.forward(
                    x,
                    x_len,
                    n_timesteps=n_timesteps,
                    temperature=1.5,
                    stoc=False,
                    spk=spk,
                    length_scale=0.91,
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Model forward pass took {elapsed_time:.4f} seconds")

                if isinstance(y_dec, list):
                    if isinstance(model, FaceTTS):
                        save_mel(y_dec[-1],f"original/mel_dec_{n_timesteps}_{elapsed_time:.2f}")
                else:
                    if isinstance(model, MyFaceTTS):
                        save_mel(y_dec,f"my/mel_dec_{n_timesteps}_{elapsed_time:.2f}")
                if isinstance(model, FaceTTS):
                        save_mel(y_enc,f"original/mel_enc_{n_timesteps}_{elapsed_time:.2f}")
                elif isinstance(model, MyFaceTTS):
                        save_mel(y_enc,f"my/mel_enc_{n_timesteps}_{elapsed_time:.2f}")
                audio = (
                    vocoder.forward(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy()
                    * 32768
                ).astype(np.int16)
                audio_encoder = (
                    vocoder.forward(y_enc).cpu().squeeze().clamp(-1, 1).numpy()
                    * 32768
                ).astype(np.int16)
                
                if isinstance(model, MyFaceTTS):
                    write(
                        f"my/sample_{i}_{n_timesteps}_{elapsed_time:.2f}.wav",
                        _config["sample_rate"],
                        audio,
                    )
                    write(
                        f"my/sample_encoder_{i}_{n_timesteps}_{elapsed_time:.2f}.wav",
                        _config["sample_rate"],
                        audio_encoder,
                    )
                else:
                    write(
                        f"original/sample_{i}_{n_timesteps}_{elapsed_time:.2f}.wav",
                        _config["sample_rate"],
                        audio,
                    )
                    write(
                        f"original/sample_encoder_{i}_{n_timesteps}_{elapsed_time:.2f}.wav",
                        _config["sample_rate"],
                        audio_encoder,
                    )

        print(f"######## Done inference. Check '{_config['output_dir']}' folder")
