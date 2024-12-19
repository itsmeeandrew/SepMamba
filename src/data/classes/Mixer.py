import glob
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import SpeedPerturbation

from src.data.classes.Preprocessor import Preprocessor

class Mixer(Dataset):
    def __init__(self, root_dir, n_samples=20000, resample_rate=8000, to_seconds=4, snr=(-2.5, 2.5), do_speed_perturb=True, subset="si_tr_s"):
        self.audio_filelist = glob.glob(root_dir + f"/wsj0/{subset}/**/*.wav", recursive=True)
        activlev_file = root_dir + f"/metadata/activlev/activlev_{subset}.txt"
        self.activlev = pd.read_csv(activlev_file, delimiter=" ", names=["filename", "activlev"], index_col="filename").to_dict()["activlev"]
        self.audio_length = to_seconds * resample_rate
        self.n_samples = n_samples
        self.snr = snr
        self.do_speed_perturb = do_speed_perturb
        self.speed_perturb = SpeedPerturbation(8000, torch.linspace(0.95, 1.05, 3))
        self.preprocessor = Preprocessor(resample_rate=resample_rate, to_length=self.audio_length)
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s1_path = random.choice(self.audio_filelist)
        s2_path = random.choice(self.audio_filelist)
        snr_dbs = [np.random.uniform(self.snr[0], 0), np.random.uniform(0, self.snr[1])]

        s1 = self.get_speaker(s1_path, snr_dbs[0])
        s2 = self.get_speaker(s2_path, snr_dbs[1])        

        if self.do_speed_perturb:
            s1,_ = self.speed_perturb(s1)
            s2,_ = self.speed_perturb(s2)

        min_length = min(s1.shape[1], s2.shape[1])
        s1 = s1[:, :min_length]
        s2 = s2[:, :min_length]

        sources = torch.stack([s1, s2], dim=0)
        mixed = torch.sum(sources, dim=0)

        # preprocess similar to min in py-wsj0
        # https://github.com/mpariente/pywsj0-mix
        gain = np.max([1., torch.max(torch.abs(mixed)), torch.max(torch.abs(sources))]) / 0.9
        mixed /= gain
        sources /= gain

        return mixed.squeeze(0), sources[0].squeeze(0), sources[1].squeeze(0)
    
    def get_speaker(self, path, snr):
        speaker, speaker_sr = torchaudio.load(path, backend="soundfile")
        speaker = self.preprocessor(speaker, speaker_sr)

        # this is the same as in py-wsj0
        activlev = self.activlev[(path.split('/')[-1]).replace(".wav", "")]
        speaker = speaker / np.sqrt(float(activlev)) * 10 ** (float(snr)/20)

        return speaker