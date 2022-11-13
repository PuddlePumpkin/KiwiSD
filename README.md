# Kiwi
Kiwi is a huggingface diffusers based discord bot for use with stable diffusion and other models

If getting errors may need to change file in torch: torch/distributed/elastic/timer/file_based_local_timer.py line 81 to `def __init__(self, file_path: str, signal=signal.SIGILL) -> None:`