<h1 align="center">
Kiwi
</h1>

<p align=center><img width="200" height="200" src="/docs/kiwipfp.png"></p><p align=center>Kiwi is a hikari lightbulb discord bot for prompting diffusion models through huggingface's diffusers library</p>


# Features
- **Models:**
Supports huggingface diffusers models and automatically converts **.ckpt** models to that format
- **Text to image:**
Enter a detailed prompt to generate an image. Optionally use a negative prompt of features to avoid
- **Image to image:**
Enter a detailed prompt and an image to generate a new one like it, strength changes the power of the input image over whats generated.
- **Inpainting:**
Enter a detailed prompt, an input image, and a black and white image mask, the white parts of the mask will be painted over with diffusion, high strength means the inpainting area will mostly be the same as the input image.
- **Textual inversion inference:**
Automatically loads TI embeds placed in the embeddings folder, must be in hugging face concepts .bin or the folder containing it from their repository, does ***not*** currently support .pt embeds because I am not smart enough to implement them.
- **Prompt weighting:**
You can multiply prompt focus in line with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**
You can reduce focus in line like negative prompts with square brackets eg: **[**1girl**]** or **[**1girl:1.3**]**  **Default: 1.1**
<p align=center><img width="700" height="323" src="/docs/examples.png"></p>

### Recommended Models:
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Waifu Diffusion v1.3](https://huggingface.co/hakurei/waifu-diffusion-v1-3)
# Usage
- With python installed, open cmd, cd to kiwi's directory, enter python -m venv venv (or python3)
- navigate to venv/scripts/activate and drag the blank activate file into cmd and press enter
- enter pip install -r requirements.txt
- Place model weights .ckpt file or the repository folder containing a diffusers format model in the kiwi/models folder, .ckpt models will take a moment to convert the first time.
- go to the discord applications page [Here](https://discord.com/developers/applications), create a new application, give it a unique name
- Go to the "bot" section -> click "add bot" -> click "reset token", this token can only be viewed once without having to reset it so take note of it. **disable public bot unless you know what you're doing**, tick the intent switches on
- Go to "OAuth2" section -> URL Generator, click bot scope -> click administrator permission, or specific permissions if you know them, copy and paste generated link into your browser or message it to who has permission to invite the bot to your discord.
- paste your token into the bottoken field of kiwitoken.json *or* set a kiwitoken environment variable to the token (on windows, open cmd, open kiwi/venv/scripts/, drag the blank activate file into cmd and press enter, enter "set kiwitoken=YOURBOTTOKEN".)
- copy your user id to the AdminList field of kiwiconfig.json or kiwiconfigdefault.json to allow you access to change **/adminsettings** options (user id's can usually be accessed via right click in discord)
- start the bot with **kiwi.bat**
- Enter /changemodel and select your model to load
- Enter /generate to start prompting
# Commands
### Generation
- **/generate:** Generates a image from a detailed description, or booru tags separated by commas. optionally input an image for image to image generation, or a mask and an image for inpainting, note inpainting works poorly with default sampler, use klms instead
- **/regenerate:** Re-generates last entered prompt
- **/overgenerate:** Diffuses from last diffusion result
### Settings
- **/changemodel:** changes the loaded model, must be used after bot is started if a default model is not set
- **/togglequalityprompt:** toggles whether or not to use the user's default quality prompt.
- **/togglenegativeprompt:** toggles whether or not to use the user's default negative prompt.
- **/settings:** displays a list of settings and optionally change them
- **/adminsettings:** displays a list of admin settings and optionally changes them **your discord user id (just a bunch of numbers) must be in kiwiconfig.json to modify these.**
- **/adminupdatecommands:** refreshes commands
### Other
- **/help:** displays command list
- **/imagetocommand:** takes an input image  or image link / message id and gives 
- **/metadata:** displays image metadata of an input image or image link / message id
- **/styles:** displays a list of loaded textual inversion embeddings
- **/styleinfo:** displays the training images of a TI if they are in the concept_images folder of the embed
# Credits
This project properly abides by all exteral package licenses and makes use of multiple third party resources. Kiwi's non derivative source and contents are licensed under the MIT license.
Largely used packages:
|Package|Link|License|
|----|----|----|
|Kiwi|https://github.com/KeiraTheCat/KiwiSD|MIT|
|Hugging Face Diffusers (Modified conversion script)|https://github.com/huggingface/diffusers|Apache 2.0|
|Gidole Font|https://github.com/larsenwork/Gidole/|MIT|
|Hikari|https://github.com/hikari-py/hikari/|MIT|
|Hikari-Lightbulb|https://github.com/tandemdude/hikari-lightbulb|LGPLv3|
See requirements.txt to view other used packages and find their licenses.

If getting errors may need to change file in torch: `venv/lib/site-packages/torch/distributed/elastic/timer/file_based_local_timer.py` line 81 to `def __init__(self, file_path: str, signal=signal.SIGILL) -> None:`