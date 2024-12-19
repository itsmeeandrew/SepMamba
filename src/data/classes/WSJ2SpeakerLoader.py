import os
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import random
from src.data.classes.Preprocessor import Preprocessor

class WSJ0Mix2SpeakerDataset(Dataset):
    def __init__(self, root_dir, subset='tr', to_length=4, process=True):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            subset (string): One of 'cv', 'tr', or 'tt' to denote the subset of data.
            to_length (int): Length of returned audios in seconds.
        """
        self.root_dir = root_dir
        self.process = process
        self.subset = subset
        self.to_length = to_length * 8000
        self.mix_dir = os.path.join(root_dir, subset, 'mix')
        self.s1_dir = os.path.join(root_dir, subset, 's1')
        self.s2_dir = os.path.join(root_dir, subset, 's2')    

        self.preprocessor = Preprocessor(8000)
        self.file_names = os.listdir(self.mix_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.mix_dir, self.file_names[idx])
        s1_path = os.path.join(self.s1_dir, self.file_names[idx])
        s2_path = os.path.join(self.s2_dir, self.file_names[idx])

        mix, _ = sf.read(mix_path, dtype='float32')
        s1, _ = sf.read(s1_path, dtype='float32')
        s2, _ = sf.read(s2_path, dtype='float32')

        # Convert to tensors
        mix = torch.tensor(mix, dtype=torch.float32)
        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)

        # Cut to equal length
        audio_length = mix.shape[0]

        if self.process:
            if audio_length > self.to_length:
                start_idx = random.randint(0, audio_length - self.to_length - 1)

                mix = mix[start_idx:(start_idx+self.to_length)]
                s1 = s1[start_idx:(start_idx+self.to_length)]
                s2 = s2[start_idx:(start_idx+self.to_length)]
            else:
                zero_pad = torch.full([self.to_length - audio_length], 0)

                mix = torch.cat((mix, zero_pad))
                s1 = torch.cat((s1, zero_pad))
                s2 = torch.cat((s2, zero_pad))

            mix = self.preprocessor.clear_nans(mix)
            s1 = self.preprocessor.clear_nans(s1)
            s2 = self.preprocessor.clear_nans(s2)

        return mix, s1, s2
