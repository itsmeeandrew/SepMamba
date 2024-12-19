import torch
import numpy as np
import hydra
import logging
import torchinfo
from omegaconf import OmegaConf
from hydra.utils import instantiate

from logger_.MetricLogger import AudioMetricsLogger
from src.utils.losses import calc_losses, pit_reorder_batch, si_sdri

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(data, model):
    mix, s1, s2 = (
        data[0].to(device, non_blocking=True),
        data[1].to(device, non_blocking=True),
        data[2].to(device, non_blocking=True),
    )
    out = model.forward(mix.unsqueeze(1))
    s1, s2 = s1[:, : out.shape[2]], s2[:, : out.shape[2]]
    sources = torch.stack([s1, s2], dim=1)
    mix = mix[:, : out.shape[2]]
    return out, sources, mix

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def train(config):
    # Print configuration
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

    # Model, optimizer and scheduler
    model = instantiate(config.model).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    train_dataset = instantiate(config.dataset)
    train_loader = instantiate(config.dataloader, dataset=train_dataset)
    val_dataset = instantiate(config.val_dataset)
    val_loader = instantiate(config.dataloader, dataset=val_dataset)

    start_epoch = 0
    if config.load.load_checkpoint:
        checkpoint = torch.load(config.load.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if config.load.load_epoch:
            start_epoch = checkpoint["epoch"]

    metric_logger = AudioMetricsLogger(
        project_name=config.wandb.project_name,
        experiment_name=config.wandb.experiment_name,
        model=model,
        cfg=config,
        output_dir=output_dir,
        use_wandb=config.wandb.use_wandb,
        log_gradients=config.wandb.log_gradients
    )
    torchinfo.summary(model)
    
    highest_si_sdri = float('-inf')
    lowest_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.epochs):
        running_loss = 0.0
        train_losses = []
        step_idx = 0
        log.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        for i, data in enumerate(train_loader):
            model.train()

            optimizer.zero_grad()
            gens, src, mix = step(data, model)
            gens = pit_reorder_batch(gens, src)
            loss = calc_losses(gens, src)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()

            # Print statistics
            if (i+1) % config.logging.print_freq == 0:
                avg_loss = running_loss / config.logging.print_freq
                step_idx = i + epoch * len(train_loader)
                metric_logger.log_lr(optimizer.param_groups[0]["lr"], step=step_idx)

                if avg_loss is None:
                    continue
                
                log.info(
                    f"[TRAIN] Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

                train_losses.append(avg_loss)
                
                metric_logger.log_audio_metrics(
                    config.wandb,
                    mix,
                    gens,
                    src,
                    sample_rate=config.dataset.resample_rate,
                    step=step_idx,
                    mode="Training",
                )
        scheduler.step()
        metric_logger.log_train_loss(np.mean(train_losses), step=step_idx)

        if epoch % config.logging.eval_freq == 0 and epoch != start_epoch:
            with torch.no_grad():
                running_loss = 0.0
                running_si_sdri = 0.0
                for i, data in enumerate(val_loader):
                    model.eval()

                    gens, src, mix = step(data, model)
                    gens = pit_reorder_batch(gens, src)
                    loss = calc_losses(src, gens)
                    
                    running_loss += loss.item()
                    running_si_sdri += si_sdri(gens, src, mix).mean()

                val_loss = running_loss / len(val_loader)
                
                avg_si_sdri = running_si_sdri / len(val_loader)

                log.info(f'[VAL] Epoch [{epoch+1}/{config.epochs}], Loss: {val_loss:.4f}')
                log.info(f'[VAL] Epoch [{epoch+1}/{config.epochs}], SI-SDRi: {avg_si_sdri:.4f}')
                
                if highest_si_sdri < avg_si_sdri:
                    highest_si_sdri = avg_si_sdri
                    metric_logger.save_model_optional_log(
                        scheduler=scheduler, 
                        epoch=epoch, 
                        optimizer=optimizer, 
                        model=model, 
                        name=f"best_model", 
                        weights_path=f"{output_dir}", 
                        log=False
                    )
                    log.info("Best model checkpoint saved.")


                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
        
                metric_logger.log_val_loss(val_loss, step=step_idx)
                metric_logger.log_si_sdri(avg_si_sdri, step=step_idx)

                log.info(f"Highest SI-SDRi: {highest_si_sdri:.4f}")
                log.info(f"Lowest validation loss: {lowest_val_loss:.4f}")

                metric_logger.save_model_optional_log(
                    scheduler=scheduler, 
                    epoch=epoch, 
                    optimizer=optimizer, 
                    model=model, 
                    name=f"last_model", 
                    weights_path=f"{output_dir}", 
                    log=False
                )

    print('Finished Training')


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt as ki:
        log.info("Training interrupted by user.")
