import math
import torch
import numpy as np
from diffusers import DDPMScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddpm import rescale_zero_terminal_snr


class SNR_DDPMScheduler(DDPMScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        trained_betas = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        snr_min=0.01, 
        snr_max=1000.0,
        snr_power = 1.0
    ):
        
        self.betas = snr_based_beta_schedule(
                num_train_timesteps,
                snr_max=snr_max,
                snr_min=snr_min,
                snr_power=snr_power,
            )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        self.variance_type = variance_type


def snr_based_beta_schedule(
    timesteps: int,
    snr_min: float = 0.01,
    snr_max: float = 1000,
    snr_power: float = 1.0,
):
    """Create beta schedule that targets specific SNR values"""
    # Create normalized time steps and apply power transformation
    t = torch.linspace(0, 1, timesteps)
    t_transformed = torch.pow(t, snr_power)

    # Create log-SNR schedule with transformed time
    log_snr_max = math.log(snr_max)
    log_snr_min = math.log(snr_min)
    log_snr = log_snr_max + (log_snr_min - log_snr_max) * t_transformed

    # Convert to SNR values
    target_snr = torch.exp(log_snr)

    # SNR = alphas_cumprod / (1 - alphas_cumprod)
    # Solve for alphas_cumprod
    alphas_cumprod = target_snr / (1 + target_snr)

    # Back out betas
    alphas = torch.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    for i in range(1, len(alphas)):
        alphas[i] = alphas_cumprod[i] / alphas_cumprod[i - 1]

    betas = 1 - alphas
    return betas.clamp(0, 0.999)
