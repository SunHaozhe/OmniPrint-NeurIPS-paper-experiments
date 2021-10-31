import numpy as np
import scipy.stats
import torch


def count_trainable_parameters(model):
    return sum([x.numel() for x in model.parameters() if x.requires_grad])


def get_torch_gpu_environment():
    env_info = dict()
    env_info["PyTorch_version"] = torch.__version__

    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["cuDNN_version"] = torch.backends.cudnn.version()
        env_info["nb_available_GPUs"] = torch.cuda.device_count()
        env_info["current_GPU_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        env_info["nb_available_GPUs"] = 0
    return env_info


def compute_mean_and_confidence_interval(x, confidence=0.95):
    """
    returns the mean and the confidence interval, which are two real numbers

    x: iterable

    high - mean_ == mean_ - low except some numerical errors in rare cases
    """
    mean_ = np.mean(x)
    low, high = scipy.stats.t.interval(confidence, len(x) - 1, loc=mean_, scale=scipy.stats.sem(x))
    return mean_, high - mean_
