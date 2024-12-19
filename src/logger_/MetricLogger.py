import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import torch.nn.functional as F
import omegaconf
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio, signal_distortion_ratio, scale_invariant_signal_distortion_ratio

from src.utils.losses import si_sdri

class AudioMetricsLogger:
    def __init__(self, project_name, experiment_name, model, cfg, output_dir, use_wandb=True, log_gradients=False):
        config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if use_wandb:
            mode = "online"
        elif not use_wandb:
            mode = "disabled"
        self.run = wandb.init(project=project_name, name=experiment_name, config=config, dir=output_dir, mode=mode)
        if log_gradients:
            self.run.watch(model, log='all')

    def log_audio_metrics(self, config, mix, generated, source, sample_rate, step, mode = "Train"):
        audio_metrics = self.compute_audio_metrics(generated, source, mix)

        if config.log_metrics:
            self.log_metrics_to_wandb(audio_metrics, step)

        mix, generated, source = self.convert_audios_to_numpy([mix, generated, source])

        # Log input audio
        if config.log_audios:
            self.log_audio(f"{mode} Input Audio", mix[0], sample_rate, step)

            # Log separated output audios
            self.log_audio(f"{mode} Output 1 Audio", generated[0][0], sample_rate, step)
            self.log_audio(f"{mode} Expected 1 Audio", source[0][0], sample_rate, step)
            self.log_audio(f"{mode} Output 2 Audio", generated[0][1], sample_rate, step)
            self.log_audio(f"{mode} Expected 2 Audio", source[0][1], sample_rate, step)

        # Compute and log spectrograms
        if config.log_spectrograms:
            self.log_spectrogram(f"{mode} Input Spectrogram", mix[0], sample_rate, step)

            self.log_spectrogram(f"{mode} Output 1 Spectrogram", generated[0][0], sample_rate, step)
            self.log_spectrogram(f"{mode} Expected 1 Spectrogram", source[0][0], sample_rate, step)

            self.log_spectrogram(f"{mode} Output 2 Spectrogram", generated[0][1], sample_rate, step)
            self.log_spectrogram(f"{mode} Expected 2 Spectrogram", source[0][1], sample_rate, step)

        # Compute and log spectral loss
        #spectral_loss_1 = self.compute_spectral_loss(output1_audio, expected1_audio)
        #spectral_loss_2 = self.compute_spectral_loss(output2_audio, expected2_audio)
        #self.run.log({"Spectral Loss Output 1": spectral_loss_1, "Spectral Loss Output 2": spectral_loss_2})

    def convert_audio_to_numpy(self, audio):
        return audio.detach().cpu().numpy()

    def convert_audios_to_numpy(self, audios):
        if len(audios[0].shape) == 3:
            audios = [audio[0] for audio in audios]
        return [self.convert_audio_to_numpy(audio) for audio in audios]

    def log_lr(self, lr, step):
         self.run.log({"Current lr": lr}, step=step)

    def log_train_loss(self, loss, step):
        self.run.log({"Train loss": loss}, step=step)

    def log_val_loss(self, loss, step):
        self.run.log({"Val loss": loss}, step=step)

    def log_si_sdri(self, si_sdri, step):
        self.run.log({"SI-SDRi": si_sdri}, step=step)

    def log_audio(self, name, audio, sample_rate, step):
        self.run.log({name: wandb.Audio(audio, sample_rate=sample_rate)}, step=step)

    def log_metrics_to_wandb(self, metrics, step):
        self.run.log(metrics, step=step)

    def log_spectrogram(self, name, audio, sample_rate, step):
        EPSILON = 1e-10
        frequencies, times, spectrogram = signal.spectrogram(audio, fs=sample_rate)
        spectrogram = np.maximum(spectrogram, EPSILON)

        plt.figure(figsize=(10, 4))
        plt.imshow(np.log(spectrogram), aspect='auto', origin='lower', 
                   extent=[times.min(), times.max(), frequencies.min(), frequencies.max()])
        plt.title(name)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.colorbar(label='Log power')
        plt.tight_layout()
        self.run.log({name: wandb.Image(plt)}, step=step)
        plt.close()

    def save_model_optional_log(self, scheduler, epoch, optimizer, model, name, weights_path="./outputs", log = False):

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            }, f"{weights_path}/{name}.pth")
        if log:
            self.run.log_model(path=weights_path,name=name)

    def compute_audio_metrics(self, gens, src, mix):
        return {
            "SI_SNR": scale_invariant_signal_noise_ratio(gens, src).mean(),
            "SDR": signal_distortion_ratio(gens, src).mean(),
            "SI_SDR": scale_invariant_signal_distortion_ratio(gens, src).mean(),
            "SI_SDRi": si_sdri(gens, src, mix).mean(),
            "MSE": F.mse_loss(gens, src).mean(),
        }

    def calculate_MCD(self, output_audio, expected_audio):
        mcd_value = self.mcd_toolbox.average_mcd(output_audio, expected_audio)
        return mcd_value

    def finish(self):
        self.run.finish()

