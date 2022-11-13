# Kiwi
![](/docs/kiwipfp.png)
Kiwi is a huggingface diffusers based discord bot for use with stable diffusion and other models

## Commands
### Generation
- **/generate:** Generates a image from a detailed description, or booru tags separated by commas. optionally input an image for image to image generation, or a mask and an image for inpainting, note inpainting works poorly with default sampler, use klms instead
- **/regenerate:** Re-generates last entered prompt
- **/overgenerate:** Diffuses from last diffusion result
### Settings
- **/changemodel:** changes the loaded model, must be used after bot is started if a default model is not set
- **/settings:** displays a list of settings and optionally change them
- **/adminsettings:** displays a list of admin settings and optionally changes them **your discord user id (just a bunch of numbers) must be in kiwiconfig.json to modify these.**
- **/adminupdatecommands:** refreshes commands
### Other
- **/help:** displays command list
- **/imagetocommand:** takes an input image  or image link / message id and gives 
- **/metadata:** displays image metadata of an input image or image link / message id
- **/styles:** displays a list of loaded textual inversion embeddings
- **/styleinfo:** displays the training images of a TI if they are in the concept_images folder of the embed
## Features
- **Models:**
Supports huggingface diffusers models and automatically converts **.ckpt** models to that format
- **Text to image:**
Enter a detailed prompt to generate an image.

- **Image to image:**
Enter a detailed prompt and an image to generate a new one like it, strength changes the power of the input image over whats generated.
- **Inpainting:**
Enter a detailed prompt, an input image, and a black and white image mask, the white parts of the mask will be painted over with diffusion, high strength means the inpainting area will mostly be the same as the input image.
- **Textual inversion inference:**
Automatically loads TI embeds placed in the embeddings folder, must be in hugging face concepts .bin or the folder containing it from their repository, does *not* currently support .pt embeds because I am not smart enough to implement them.
- **Prompt weighting:**
You can multiply prompt focus with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**
You can reduce focus in line like negative prompts with square brackets eg: **[**1girl**]** or **[**1girl:1.3**]**  **Default: 1.1**

If getting errors may need to change file in torch: `torch/distributed/elastic/timer/file_based_local_timer.py` line 81 to `def __init__(self, file_path: str, signal=signal.SIGILL) -> None:`