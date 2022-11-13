# Kiwi
Kiwi is a huggingface diffusers based discord bot for use with stable diffusion and other models

## Commands
**Generation**
**/generate**: Generates a image from a detailed description, or booru tags separated by commas"
**/regenerate**: Re-generates last entered prompt"
**/overgenerate**: Diffuses from last diffusion result"
**Settings**
**/changemodel**: switches model between stable diffusion v1.5, waifu diffusion v1.3, and yabai diffusion v???"
**/settings**: displays a list of settings and optionally change them"
**Other**
**/styles**: displays a list of loaded textual inversions"
**/styleinfo**: displays the training images of a TI"

## Features
Prompt weighting:
You can multiply prompt focus with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**"
You can reduce focus in line like negative prompts with square brackets eg: **[**1girl**]** or **[**1girl:1.3**]**  **Default: 1.1** "

If getting errors may need to change file in torch: torch/distributed/elastic/timer/file_based_local_timer.py line 81 to `def __init__(self, file_path: str, signal=signal.SIGILL) -> None:`