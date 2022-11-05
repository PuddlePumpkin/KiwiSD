# Kiwi
 Kiwi diffusers discord bot
If getting errors may need to change file in torch: torch/distributed/elastic/timer/file_based_local_timer.py line 81 to `def __init__(self, file_path: str, signal=signal.SIGILL) -> None:`
If needing to convert spicy ai models to diffusers use diffusers github source diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py **BUT** change line 629: `text_model.load_state_dict(text_model_dict)` to `text_model.load_state_dict(text_model_dict, strict=False)`