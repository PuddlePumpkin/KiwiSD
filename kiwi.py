import asyncio
import gc
import json
import os
import random
import shutil
import sys
import traceback
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import hikari
import lightbulb
import requests
import torch
from lightbulb.ext import tasks
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer

import convertckpt
from diffusers import (DDIMScheduler, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler)

# ----------------------------------
# Instantiate a Bot instance
# ----------------------------------
os.chdir(str(os.path.abspath(os.path.dirname(__file__))))
bottoken = ""
with open('kiwitoken.json', 'r') as openfile:
    tokendict = json.load(openfile)
    if tokendict["bottoken"] != "" and tokendict["bottoken"] != None:
        bottoken = tokendict["bottoken"]
    else:
        try:
            bottoken = os.environ["kiwitoken"]
        except:
            pass
    openfile.close()
if bottoken == None or bottoken == "":
    sys.exit("\nYou need a bot token, see readme.md for usage instructions")
bot = lightbulb.BotApp(token=bottoken,prefix="-",default_enabled_guilds=672718048343490581,intents=hikari.Intents.ALL,help_class=None)

# ----------------------------------
# Classes
# ----------------------------------
class imageRequest(object):
    def __init__(self, Prompt=None, NegativePrompt=None, InfSteps=None, Seed=None, GuideScale=None, ImgUrl=None, Strength=None, Width=None, Height=None, Proxy=None, resultImage=None, regenerate=False, overProcess=False, scheduler=None, userconfig=None, author=None, InpaintUrl=None, isAnimation=False, gifFrame=None, doLabel = False, labelKey = None):
        self.prompt = Prompt
        self.negativePrompt = NegativePrompt
        self.infSteps = InfSteps
        self.seed = Seed
        self.guideScale = GuideScale
        self.imgUrl = ImgUrl
        self.inpaintUrl = InpaintUrl
        self.strength = Strength
        self.width = Width
        self.height = Height
        self.proxy = Proxy
        self.resultImage = resultImage
        self.regenerate = regenerate
        self.overProcess = overProcess
        self.scheduler = scheduler
        self.userconfig = userconfig
        self.author = author
        self.isAnimation = isAnimation
        self.gifFrame = gifFrame
        self.doLabel = doLabel
        self.labelKey = labelKey


class animationRequest(object):
    def __init__(self, Prompt=None, NegativePrompt=None, InfSteps=None, Seed=None, GuideScale=None, ImgUrl=None, Strength=None, Width=None, Height=None, Proxy=None, resultImage=None, regenerate=False, overProcess=False, scheduler=None, userconfig=None, author=None, InpaintUrl=None, isAnimation=True, startframe=None, endframe=None, animkey=None, animstep=None, ingif=None, LabelFrames=False):
        self.prompt = Prompt
        self.negativePrompt = NegativePrompt
        self.infSteps = InfSteps
        self.seed = Seed
        self.guideScale = GuideScale
        self.imgUrl = ImgUrl
        self.inpaintUrl = InpaintUrl
        self.strength = Strength
        self.width = Width
        self.height = Height
        self.proxy = Proxy
        self.labelframes = LabelFrames
        self.resultImage = resultImage
        self.regenerate = regenerate
        self.overProcess = overProcess
        self.scheduler = scheduler
        self.userconfig = userconfig
        self.author = author
        self.startframe = startframe
        self.endframe = endframe
        self.animkey = animkey
        self.animstep = animstep
        self.currentstep = startframe
        self.ingif = ingif


class threadManager(object):
    def New_Thread(self, request: imageRequest = None, previous_request: imageRequest = None):
        return genImgThreadClass(parent=self, request=request, previous_request=previous_request)

    def on_thread_finished(self, thread, data: hikari.Embed, request: imageRequest, proxy: lightbulb.ResponseProxy, author: hikari.User):
        global awaitingProxy
        global awaitingEmbed
        global previous_request
        global awaitingAuthor
        global animationFrames
        global awaitingFrame
        if request.isAnimation:
            awaitingFrame = request.resultImage
            previous_request = request
            return
        previous_request = request
        awaitingProxy = proxy
        awaitingEmbed = data
        awaitingAuthor = author


class genImgThreadClass(Thread):
    '''thread class for generation'''

    def __init__(self, parent=None, request: imageRequest = None, previous_request: imageRequest = None):
        self.parent = parent
        self.request = request
        self.previous_request = previous_request
        super(genImgThreadClass, self).__init__()

    def run(self):
        global outputDirectory
        global pipe
        global curmodel
        global botBusy
        # Handle Scheduler
        if (self.request.scheduler == None or self.request.scheduler == "0"):
            self.request.scheduler = "KLMS"
            if self.request.regenerate:
                if self.previous_request.scheduler != None:
                    self.request.scheduler = self.previous_request.scheduler

        if self.request.scheduler == "KLMS":
            scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif self.request.scheduler == "Euler":
            scheduler = EulerDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        elif self.request.scheduler == "PNDM":
            scheduler = PNDMScheduler(beta_end=0.012, beta_schedule="scaled_linear", beta_start=0.00085,
                                      num_train_timesteps=1000, set_alpha_to_one=False, skip_prk_steps=True, steps_offset=1, trained_betas=None)
        elif self.request.scheduler == "DPM++":
            scheduler = DPMSolverMultistepScheduler.from_config(curmodel["ModelPath"], subfolder="scheduler", solver_order=2, predict_epsilon=True, thresholding=False,
                                                                algorithm_type="dpmsolver++", solver_type="midpoint", denoise_final=True)  # the influence of this trick is effective for small (e.g. <=10) steps)
        pipe.scheduler = scheduler
        # Handle prompt
        if (self.request.prompt == None or self.request.prompt == "0"):
            self.request.prompt = ""
            if self.request.regenerate:
                if self.previous_request.prompt != None:
                    self.request.prompt = self.previous_request.prompt

        # Handle negative prompt
        if (self.request.negativePrompt == None or self.request.negativePrompt == "0"):
            self.request.negativePrompt = None
            if self.request.regenerate:
                if self.previous_request.negativePrompt != None:
                    self.request.negativePrompt = self.previous_request.negativePrompt

        # Handle Default Negative
        if not self.request.regenerate:
            if self.request.userconfig["UseDefaultNegativePrompt"]:
                if (self.request.negativePrompt != None):
                    if (self.request.negativePrompt == ""):
                        self.request.negativePrompt = self.request.userconfig["DefaultNegativePrompt"]
                    else:
                        self.request.negativePrompt = self.request.negativePrompt + \
                            ", " + \
                            self.request.userconfig["DefaultNegativePrompt"]
                else:
                    self.request.negativePrompt = self.request.userconfig["DefaultNegativePrompt"]

        # Handle Default Quality
        if not self.request.regenerate:
            if self.request.userconfig["UseDefaultQualityPrompt"]:
                if (self.request.prompt != None):
                    if (self.request.prompt == ""):
                        self.request.prompt = self.request.userconfig["DefaultQualityPrompt"]
                    else:
                        self.request.prompt = self.request.userconfig["DefaultQualityPrompt"] + \
                            ", " + self.request.prompt
                else:
                    self.request.prompt = self.request.userconfig["DefaultQualityPrompt"]

        # Handle infsteps
        if (self.request.infSteps == "0" or self.request.infSteps == 0 or self.request.infSteps == None):
            self.request.infSteps = 15
            if self.request.regenerate:
                if self.previous_request.infSteps != None:
                    self.request.infSteps = self.previous_request.infSteps

        # Handle guidescale
        if (self.request.guideScale == "0" or self.request.guideScale == None):
            self.request.guideScale = 7.0
            if self.request.regenerate:
                if self.previous_request.guideScale != None:
                    self.request.guideScale = self.previous_request.guideScale

        # Handle Stength
        if (self.request.strength == None):
            self.request.strength = 0.25
            if self.request.regenerate:
                if self.previous_request.strength != None:
                    self.request.strength = self.previous_request.strength

        # Handle ImgUrl
        if (self.request.imgUrl == "0" or self.request.imgUrl == None):
            self.request.imgUrl = None
            if self.request.regenerate:
                if self.previous_request.imgUrl != None:
                    self.request.imgUrl = self.previous_request.imgUrl
                elif self.request.gifFrame == None:
                    self.request.strength = None
            elif self.request.overProcess:
                pass
            elif self.request.gifFrame == None:
                self.request.strength = None

        # Handle inpaint ImgUrl
        if (self.request.inpaintUrl == "0" or self.request.inpaintUrl == None):
            self.request.inpaintUrl = None
            if self.request.regenerate:
                if self.previous_request.inpaintUrl != None:
                    self.request.inpaintUrl = self.previous_request.inpaintUrl

        # Handle Width
        if (self.request.width == 0 or self.request.width == "0" or self.request.width == None):
            self.request.width = 512
            if self.request.regenerate:
                if self.previous_request.width != None:
                    self.request.width = self.previous_request.width

        # Handle Height
        if (self.request.height == 0 or self.request.height == "0" or self.request.height == None):
            self.request.height = 512
            if self.request.regenerate:
                if self.previous_request.height != None:
                    self.request.height = self.previous_request.height

        # Handle seed
        if (self.request.seed == 0 or self.request.seed == None):
            self.request.seed = random.randint(1, 100000000)
        generator = torch.Generator("cuda").manual_seed(self.request.seed)

        # handle overprocess
        if self.request.regenerate:
            if self.previous_request.overProcess != None:
                self.request.overProcess = self.previous_request.overProcess

        # Load Image
        if self.request.imgUrl == None:
            init_image = None
        elif (not self.request.overProcess):
            print("Loading image from url: " + self.request.imgUrl)
            response = requests.get(self.request.imgUrl)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            #Crop and resize
            init_image = crop_and_resize(
                init_image, self.request.width, self.request.height)
            init_image = init_image.resize(
                (self.request.width, self.request.height), Image.Resampling.LANCZOS)
        if self.request.overProcess:
            print("Loading image from previous_request.resultImage")
            init_image = self.previous_request.resultImage
        if self.request.gifFrame != None:
            # initialize init image from gif frame
            init_image = self.request.gifFrame
            init_image = crop_and_resize(
                init_image, self.request.width, self.request.height)
            init_image = init_image.resize(
                (self.request.width, self.request.height), Image.Resampling.LANCZOS)
        # Load inpaint Image
        if self.request.inpaintUrl != None:
            inpaint_image = None
            print("Loading inpaint mask from url: " + self.request.inpaintUrl)
            response = requests.get(self.request.inpaintUrl)
            inpaint_image = Image.open(
                BytesIO(response.content)).convert("RGB")
            #Crop and resize
            inpaint_image = crop_and_resize(
                init_image, self.request.width, self.request.height)
            inpaint_image = inpaint_image.resize(
                (self.request.width, self.request.height), Image.Resampling.LANCZOS)

        # Check for duplicate tokens
        if self.request.prompt != None:
            self.request.prompt = remove_duplicates(self.request.prompt)
        print("Generating:" + self.request.prompt)
        if self.request.negativePrompt != None:
            self.request.negativePrompt = remove_duplicates(
                self.request.negativePrompt)
        # Set Metadata
        metadata = PngInfo()
        if ((self.request.prompt != None) and (self.request.prompt != "None") and (self.request.prompt != "")):
            metadata.add_text("Prompt", self.request.prompt)
        if ((self.request.negativePrompt != None) and (self.request.negativePrompt != "None") and (self.request.negativePrompt != "")):
            metadata.add_text("Negative Prompt", self.request.negativePrompt)
        metadata.add_text("Guidance Scale", str(self.request.guideScale))
        metadata.add_text("Inference Steps", str(self.request.infSteps))
        metadata.add_text("Seed", str(self.request.seed))
        metadata.add_text("Width", str(self.request.width))
        metadata.add_text("Height", str(self.request.height))
        metadata.add_text("Scheduler", str(self.request.scheduler))
        try:
            metadata.add_text("Model", curmodel["ModelDetailedName"])
        except:
            metadata.add_text("Model", curmodel["ModelCommandName"])

        # Generate
        nsfwDetected = False
        with autocast("cuda"):
            if not config["EnableNsfwFilter"]:
                def dummy_checker(images, **kwargs): return images, False
                pipe.safety_checker = dummy_checker
            if self.request.strength != None:
                if self.request.inpaintUrl == None:
                    print("Strength = " + str(1-self.request.strength))
                    returndict = pipe.img2img(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, init_image=init_image,
                                              negative_prompt=self.request.negativePrompt, strength=(1-(self.request.strength*0.89)), guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps)
                    image = returndict.images[0]
                    try:
                        nsfwDetected = returndict.nsfw_content_detected[0]
                    except:
                        nsfwDetected = False
                else:
                    returndict = pipe.inpaint(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, init_image=init_image, negative_prompt=self.request.negativePrompt, strength=(
                        1-(self.request.strength*0.89)), mask_image=inpaint_image, guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps)
                    image = returndict.images[0]
                    try:
                        nsfwDetected = returndict.nsfw_content_detected[0]
                    except:
                        nsfwDetected = False
                metadata.add_text("Img2Img Strength",
                                  str(self.request.strength))
            else:
                returndict = pipe.text2img(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, init_image=init_image,
                                           negative_prompt=self.request.negativePrompt, guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps)
                image = returndict.images[0]
                try:
                    nsfwDetected = returndict.nsfw_content_detected[0]
                except:
                    nsfwDetected = False
        countStr = str(filecount()+1)
        while os.path.exists(outputDirectory + str(countStr) + ".png"):
            countStr = int(countStr)+1

        # Process Result
        self.request.resultImage = image
        image.save(outputDirectory + str(countStr) + ".png", pnginfo=metadata)
        if not nsfwDetected:
            outEmbed = get_embed(self.request.prompt, self.request.negativePrompt, self.request.guideScale, self.request.infSteps, self.request.seed,
                                 outputDirectory + str(countStr) + ".png", self.request.strength, False, self.request.scheduler, self.request.userconfig, self.request.imgUrl)
        else:
            outEmbed = hikari.Embed(title=config["NsfwMessage"], colour=hikari.Colour(0xFF0000)).set_footer(
                "An admin may enable possible nsfw results in /adminsettings... sometimes the detector finds nsfw in sfw results")
        self.parent and self.parent.on_thread_finished(
            self, outEmbed, self.request, self.request.proxy, self.request.author)


# ----------------------------------
# Configs
# ----------------------------------
def load_config():
    '''loads admin config file'''
    global config
    if not os.path.exists(str("kiwiconfig.json")):
        shutil.copy2("kiwiconfigdefault.json", "kiwiconfig.json")
    with open('kiwiconfig.json', 'r') as openfile:
        config = json.load(openfile)
        openfile.close()


def save_config():
    '''Saves admin config file'''
    global config
    with open("kiwiconfig.json", "w") as outfile:
        json.dump(config, outfile, indent=4)
        outfile.close()


def string_to_bool(option) -> bool:
    '''Converts string to bool from various wordings'''
    try:
        false_strings = ["off", "false", "no"]
        true_strings = ["on", "true", "yes"]
        if str(option).lower().strip() in false_strings:
            return False
        elif str(option).lower().strip() in true_strings:
            return True
        else:
            return False
    except:
        return False


def get_admin_list() -> list:
    global config
    return config["AdminList"].replace(", ", "").replace(" ,", "").replace(" , ", "").split(",")


def load_user_config(userid: str) -> dict:
    '''Load a user config from the specified user id'''
    global config
    if not os.path.exists(str("usersettings.json")):
        shutil.copy2("usersettingsdefault.json", "usersettings.json")
    with open('usersettings.json', 'r') as openfile:
        userconfig = json.load(openfile)
        openfile.close()
    if userid in userconfig:
        return userconfig[userid]
    else:
        # write a default config to the userid
        load_config()
        userconfig[userid] = {"UseDefaultQualityPrompt": False, "DefaultQualityPrompt": config["NewUserQualityPrompt"],
                              "UseDefaultNegativePrompt": False, "DefaultNegativePrompt": config["NewUserNegativePrompt"]}
        return userconfig[str(userid)]


def save_user_config(userid: str, saveconfig):
    '''Saves a user setting to the json file'''
    with open('usersettings.json', 'r') as openfile:
        userconfigs = json.load(openfile)
        userconfigs[str(userid)] = saveconfig
        openfile.close()
    with open("usersettings.json", "w") as outfile:
        json.dump(userconfigs, outfile, indent=4)
        outfile.close()


# ----------------------------------
# Helpers
# ----------------------------------
def convert_ckpt(ckptpath, vaepath=None, dump_path=None):
    convertckpt.convertmodel(ckptpath, vaepath, dump_path=dump_path)


def populate_model_list():
    global model_list
    ckptlist = list(Path("./models/").rglob("*.[cC][kK][pP][tT]"))
    for ckpt in ckptlist:
        try:
            if not os.path.exists("./models/" + str(ckpt.stem) + "/"):
                if os.path.exists("./models/" + str(ckpt.stem) + ".vae.pt"):
                    print("\nConverting with vae: " +
                          str(ckpt).replace("\\", "/") + "\n")
                    convert_ckpt(str(ckpt), "./models/" + str(ckpt.stem) +
                                 ".vae.pt", dump_path="models/" + str(ckpt.stem) + "/")
                else:
                    print("\nConverting model without vae: " +
                          str(ckpt).replace("\\", "/") + "\n")
                    convert_ckpt(str(ckpt), dump_path="models/" +
                                 str(ckpt.stem) + "/")
                print("\nConversion Complete: " + str(ckpt).replace("\\", "/"))
                print("Diffusers Model Path: " +
                      "models/" + str(ckpt.stem) + "/")
        except:
            print("Failed to convert model: " + str(ckpt))
    for folder in next(os.walk('./models'))[1]:
        json_list = list(
            Path("./models/"+folder).rglob("*[mM][oO][dD][eE][lL].[jJ][sS][oO][nN]"))
        if len(json_list) == 0:
            model_list[folder] = {"ModelCommandName": folder,
                                  "ModelPath": "./models/" + folder, "ModelDetailedName": folder}
        else:
            with open(json_list[0], 'r') as openfile:
                open_json = json.load(openfile)
                # print(open_json["ModelCommandName"])
                open_json["ModelPath"] = "./models/" + folder
                model_list[open_json["ModelCommandName"]] = open_json
                openfile.close()


# ----------------------------------
# load embeddings Function
# ----------------------------------
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    trained_token = list(loaded_learned_embeds.keys())[0]
    embedsS = loaded_learned_embeds[trained_token]
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embedsS.to(dtype)
    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
    print(token)
    # resize the token embeddings
    if not 'string_to_param' in loaded_learned_embeds:
        text_encoder.resize_token_embeddings(len(tokenizer))
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embedsS


# ----------------------------------
# Change Pipeline Function
# ----------------------------------
def change_pipeline(modelname):
    global pipe
    global curmodel
    global tokenizer
    global text_encoder
    global model_list
    try:
        del tokenizer
        del text_encoder
    except:
        pass
    print("\nChanging model to: " + modelname)
    tokenizer = CLIPTokenizer.from_pretrained(
        model_list[modelname]["ModelPath"], subfolder="tokenizer", use_auth_token="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")
    text_encoder = CLIPTextModel.from_pretrained(
        model_list[modelname]["ModelPath"], subfolder="text_encoder", use_auth_token="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", torch_dtype=torch.float16)
    print("\nLoading Embeds...")
    for file in embedlist:
        load_learned_embed_in_clip(str(file), text_encoder, tokenizer)
    curmodel = model_list[modelname]
    try:
        del pipe
    except:
        pass
    gc.collect()
    pipe = DiffusionPipeline.from_pretrained(model_list[modelname]["ModelPath"], custom_pipeline="lpw_stable_diffusion", use_auth_token="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",
                                             torch_dtype=torch.float16, revision="fp16", text_encoder=text_encoder, tokenizer=tokenizer, device_map="auto").to('cuda')
    print(modelname + " loaded.\n")
    pipe.enable_attention_slicing()


# ----------------------------------
# Filecount Function
# ----------------------------------
def filecount():
    global outputDirectory
    return len([entry for entry in os.listdir(outputDirectory) if os.path.isfile(os.path.join(outputDirectory, entry))])


# ----------------------------------
# isurl Function
# ----------------------------------
def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# ----------------------------------
# Image Helpers
# ----------------------------------
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def crop_and_resize(pil_img: Image.Image, width, height) -> Image.Image:
    imwidth, imheight = pil_img.size
    pilcopy = pil_img
    if imwidth > width and imheight > height:
        pilcopy.thumbnail((width, height), Image.Resampling.LANCZOS)
    imwidth, imheight = pilcopy.size
    if imwidth >= width:
        crop_center(pilcopy, width, imheight)
    imwidth, imheight = pilcopy.size
    if imheight >= height:
        crop_center(pilcopy, imwidth, imheight)
    imwidth, imheight = pilcopy.size
    return pilcopy


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


# ----------------------------------
# Get Embed
# ----------------------------------
def get_embed(Prompt, NegativePrompt: str, GuideScale, InfSteps, Seed, File, ImageStrength=None, Gifmode=False, Scheduler=None, UserConfig=None, imgurl=None):
    global curmodel
    global config
    if (not config["ShowDefaultPrompts"]):
        if (UserConfig["UseDefaultNegativePrompt"]):
            if NegativePrompt != None:
                defaultClipped = remove_duplicates(
                    UserConfig["DefaultNegativePrompt"])
                clippedList = defaultClipped.split(", ")
                NegativePrompt = remove_duplicates(NegativePrompt)
                NegativePromptList = NegativePrompt.split(", ")
                res = [i for i in NegativePromptList if i not in clippedList]
                working = set()
                result = []
                for item in res:
                    if item not in working:
                        working.add(item)
                        result.append(item)
                s = ", "
                NegativePrompt = s.join(result)
        if (UserConfig["UseDefaultQualityPrompt"]):
            if Prompt != None:
                defaultClipped = remove_duplicates(
                    UserConfig["DefaultQualityPrompt"])
                clippedList = defaultClipped.split(", ")
                Prompt = remove_duplicates(Prompt)
                PromptList = Prompt.split(", ")
                res = [i for i in PromptList if i not in clippedList]
                working = set()
                result = []
                for item in res:
                    if item not in working:
                        working.add(item)
                        result.append(item)
                s = ", "
                Prompt = s.join(result)

    if ImageStrength != None:
        footer = ("{:30s}{:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:30s}{:30s}".format(
            "Seed: "+str(Seed), "Sampler: " + str(Scheduler)) + "\n" + "{:30s}".format("Image Strength: " + str(ImageStrength)))
    else:
        footer = ("{:30s}{:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: " +
                  str(InfSteps))+"\n"+"{:30s}{:30s}".format("Seed: "+str(Seed), "Sampler: " + str(Scheduler)))
    f = hikari.File(File)
    try:
        embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(
            0x56aaf8)).set_footer(text=footer, icon=curmodel["ModelThumbnail"]).set_image(f)
    except:
        embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(
            0x56aaf8)).set_footer(text=footer).set_image(f)
    try:
        embed.title = curmodel["ModelDetailedName"] + " - Result:"
    except:
        embed.title = curmodel["ModelCommandName"] + " - Result:"
    try:
        embed.color = hikari.Colour(int(curmodel["ModelColor"], 16))
    except:
        embed.color = hikari.Colour(0xff8f87)
    if imgurl != None:
        embed.set_thumbnail(imgurl)
    if ((Prompt != None) and (Prompt != "None") and (Prompt != "")):
        embed.add_field("Prompt:", Prompt)
    if ((NegativePrompt != None) and (NegativePrompt != "None") and (NegativePrompt != "")):
        embed.add_field("Negative Prompt:", NegativePrompt)
    if (Gifmode):
        embed.set_footer(None)
        embed.set_image(File)
    return embed


# ----------------------------------
# Button Functions
# ----------------------------------
async def generate_rows(bot: lightbulb.BotApp):
    rows = []
    row = bot.rest.build_action_row()
    label = "🗑️"
    row.add_button(hikari.ButtonStyle.SECONDARY,
                   "delete").set_label(label).add_to_container()
    rows.append(row)
    return rows


async def generate_toggle_rows(bot: lightbulb.BotApp, quality: bool):
    rows = []
    row = bot.rest.build_action_row()
    label = "🗑️"
    label1 = "🟢"
    if not quality:
        row.add_button(hikari.ButtonStyle.SECONDARY, "toggle").set_label(
            label1).add_to_container()
    else:
        row.add_button(hikari.ButtonStyle.SECONDARY, "togglequality").set_label(
            label1).add_to_container()
    row.add_button(hikari.ButtonStyle.SECONDARY,
                   "delete").set_label(label).add_to_container()
    rows.append(row)
    return rows


async def handle_responses(bot: lightbulb.BotApp, author: hikari.User, message: hikari.Message, ctx: lightbulb.SlashContext = None, autodelete: bool = False) -> None:
    """Watches for events, and handles responding to them."""
    with bot.stream(hikari.InteractionCreateEvent, 60).filter(lambda e: (isinstance(e.interaction, hikari.ComponentInteraction) and e.interaction.user == author and e.interaction.message == message)) as stream:
        async for event in stream:
            cid = event.interaction.custom_id
            if cid == "toggle":
                embed = togprompts(False, ctx)
            elif cid == "togglequality":
                embed = togprompts(True, ctx)
            elif cid == "delete":
                await bot.rest.delete_message(message.channel_id, message)
                return
            try:
                await event.interaction.create_initial_response(hikari.ResponseType.MESSAGE_UPDATE, embed=embed)
            except hikari.NotFoundError:
                await event.interaction.edit_initial_response(embed=embed)
    # 1 minute, remove buttons
    if autodelete:
        await bot.rest.delete_message(message.channel_id, message)
    else:
        await message.edit(components=[])


# ----------------------------------
# Duplicate token handler
# ----------------------------------
def remove_duplicates(string: str) -> str:
    separated = string.replace(", ", ",")
    separated = separated.strip()
    separated = separated.replace(" ,", ",")
    separated = separated.replace(" , ", ",")
    separated = separated.split(",")

    working = set()
    result = []
    for item in separated:
        if item not in working:
            working.add(item)
            result.append(item)
    s = ", "
    return s.join(result)


# ----------------------------------
# Globals
# ----------------------------------
config = {}
model_list = {}
embedlist = []
animationFrames = []
previous_request = imageRequest()
genThread = Thread()
botBusy = False
startbool = False
awaitingEmbed = None
loadedgif = None
awaitingProxy = None
awaitingAuthor = None
awaitingFrame = None
activeAnimRequest = None
curmodel = ""
embedlist = list(Path("./embeddings/").rglob("*.[bB][iI][nN]"))
titles = ["I'll try to make that for you!...", "Maybe I could make that...",
          "I'll try my best!...", "This might be tricky to make..."]
regentitles = ["I'll try again!... ", "Sorry if I didn't do good enough... ",
               "I'll try my best to do better... "]
outputDirectory = "./results/"
# ----------------------------------
# Setup
# ----------------------------------
def setup():
    populate_model_list()
    if config["AutoLoadedModel"] != "None":
        if config["AutoLoadedModel"] in model_list:
            change_pipeline(config["AutoLoadedModel"])
        else:
            print("Auto loaded model not found...")

# ----------------------------------
# Ping Command
# ----------------------------------
@bot.command
@lightbulb.command("ping", "checks the bot is alive")
@lightbulb.implements(lightbulb.SlashCommand)
async def ping(ctx: lightbulb.SlashContext) -> None:
    await respond_with_autodelete("Pong!", ctx, 0x00ff1a)


# ----------------------------------
# Metadata Command
# ----------------------------------
@bot.command
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.option("image_link", "image link or message ID", required=False, type=str)
@lightbulb.command("metadata", "check metadata of an image")
@lightbulb.implements(lightbulb.SlashCommand)
async def metadata(ctx: lightbulb.SlashContext) -> None:
    if (ctx.options.image != None):
        datas = await hikari.Attachment.read(ctx.options.image)
        mdataimage = Image.open(BytesIO(datas)).convert("RGB")
        mdataimage = mdataimage.resize((512, 512))
        url = ctx.options.image.url
    elif (ctx.options.image_link != None):
        if (is_url(ctx.options.image_link)):
            response = requests.get(ctx.options.image_link)
            url = ctx.options.image_link
            mdataimage = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id, ctx.options.image_link)
            datas = await hikari.Attachment.read(messageIdResponse.embeds[0].image)
            mdataimage = Image.open(BytesIO(datas)).convert("RGB")
            mdataimage = mdataimage.resize((512, 512))
            url = messageIdResponse.embeds[0].image.url
        mdataimage = mdataimage.resize((512, 512))
    embed = hikari.Embed(title=(url.rsplit(
        '/', 1)[-1]), colour=hikari.Colour(0x56aaf8)).set_thumbnail(url)
    if (str(mdataimage.info.get("Model")) != "None"):
        embed.add_field("Model:", str(mdataimage.info.get("Model")))
    if (str(mdataimage.info.get("Prompt")) != "None"):
        embed.add_field("Prompt:", str(mdataimage.info.get("Prompt")))
    if (str(mdataimage.info.get("Negative Prompt")) != "None"):
        embed.add_field("Negative Prompt:", str(
            mdataimage.info.get("Negative Prompt")))
    if (str(mdataimage.info.get("Guidance Scale")) != "None"):
        embed.add_field("Guidance Scale:", str(
            mdataimage.info.get("Guidance Scale")))
    if (str(mdataimage.info.get("Inference Steps")) != "None"):
        embed.add_field("Inference Steps:", str(
            mdataimage.info.get("Inference Steps")))
    if (str(mdataimage.info.get("Seed")) != "None"):
        embed.add_field("Seed:", str(mdataimage.info.get("Seed")))
    if (str(mdataimage.info.get("Width")) != "None"):
        embed.add_field("Width:", str(mdataimage.info.get("Width")))
    if (str(mdataimage.info.get("Height")) != "None"):
        embed.add_field("Height:", str(mdataimage.info.get("Height")))
    if (str(mdataimage.info.get("Scheduler")) != "None"):
        embed.add_field("Scheduler:", str(mdataimage.info.get("Scheduler")))
    if (str(mdataimage.info.get("Img2Img Strength")) != "None"):
        embed.add_field("Img2Img Strength:", str(
            mdataimage.info.get("Img2Img Strength")))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=True)

load_config()
# ----------------------------------
# Thread result listener
# ----------------------------------
ThreadCompletionSpeed = 2
@tasks.task(s=ThreadCompletionSpeed, auto_start=True)
async def ThreadCompletionLoop():
    global awaitingProxy
    global awaitingEmbed
    global awaitingAuthor
    global botBusy
    global activeAnimRequest
    global animationFrames
    global awaitingFrame
    global startbool
    global ThreadCompletionSpeed
    global loadedgif
    if activeAnimRequest != None:
        ThreadCompletionSpeed = 0.5
        if activeAnimRequest.ingif != None:
            if (awaitingFrame != None):
                animationFrames.append(awaitingFrame)
            if (awaitingFrame != None or startbool):
                if loadedgif == None:
                    response = requests.get(activeAnimRequest.ingif.url)
                    loadedgif = Image.open(
                        BytesIO(response.content))  # .convert("RGB")
                try:
                    threadmanager = threadManager()
                    load_config()
                    loadedgif.seek(loadedgif.tell()+1)
                    requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.currentstep, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height, activeAnimRequest.proxy,
                                                 scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True, gifFrame=loadedgif)
                    thread = threadmanager.New_Thread(
                        requestObject, previous_request)
                    thread.start()
                    awaitingFrame = None
                    startbool = False
                    return
                except EOFError:
                    animationFrames[0].save(outputDirectory + "resultgif.gif", save_all=True,
                                            append_images=animationFrames[1:], duration=86, loop=0)
                    file_name = outputDirectory + "resultgif.gif"
                    file_stats = os.stat(file_name)
                    if ((file_stats.st_size / (1024 * 1024)) < 8):
                        print("Anim Complete, sending gif.")
                        #embed = hikari.Embed(title=("Animation Result:"),colour=hikari.Colour(0xFFFFFF)).set_image(outputDirectory + "resultgif.gif")
                        embed = get_embed(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.guideScale, activeAnimRequest.infSteps, activeAnimRequest.seed,
                                          outputDirectory + "resultgif.gif", activeAnimRequest.strength, True, activeAnimRequest.scheduler, activeAnimRequest.userconfig, activeAnimRequest.imgUrl)
                        embed.set_footer(None)
                        embed.set_image(outputDirectory + "resultgif.gif")
                        embed.set_thumbnail(None)
                        await activeAnimRequest.proxy.edit(embed)
                    else:
                        print("Anim Complete, Gif too big.")
                        embed = hikari.Embed(title=(
                            "Animation Complete. (Gif file too large for upload)"), colour=hikari.Colour(0xFFFFFF))
                        await activeAnimRequest.proxy.edit(embed)
                    activeAnimRequest = None
                    awaitingFrame = None
                    botBusy = False
                    loadedgif = None
                    animationFrames = []
                    return
        if (awaitingFrame != None):
            animationFrames.append(awaitingFrame)
        if (awaitingFrame != None or startbool):
            if activeAnimRequest.currentstep < activeAnimRequest.endframe:
                activeAnimRequest.currentstep = activeAnimRequest.currentstep + \
                    activeAnimRequest.animstep
                activeAnimRequest.animstep
                threadmanager = threadManager()
                load_config()
                if activeAnimRequest.animkey == "guidescale":
                    requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.currentstep, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                 activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True)
                if activeAnimRequest.animkey == "steps":
                    requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, int(activeAnimRequest.currentstep), activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                 activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True)
                if activeAnimRequest.animkey == "strength":
                    requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.currentstep, activeAnimRequest.width, activeAnimRequest.height,
                                                 activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True)
                thread = threadmanager.New_Thread(
                    requestObject, previous_request)
                thread.start()
                awaitingFrame = None
                startbool = False
            else:
                animationFrames[0].save(outputDirectory + "resultgif.gif", save_all=True,
                                        append_images=animationFrames[1:], duration=86, loop=0)
                file_name = outputDirectory + "resultgif.gif"
                file_stats = os.stat(file_name)
                if ((file_stats.st_size / (1024 * 1024)) < 8):
                    print("Anim Complete, sending gif.")
                    #embed = hikari.Embed(title=("Animation Result:"),colour=hikari.Colour(0xFFFFFF)).set_image(outputDirectory + "resultgif.gif")
                    embed = get_embed(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.guideScale, activeAnimRequest.infSteps, activeAnimRequest.seed,
                                      outputDirectory + "resultgif.gif", activeAnimRequest.strength, True, activeAnimRequest.scheduler, activeAnimRequest.userconfig, activeAnimRequest.imgUrl)
                    embed.set_footer(None)
                    embed.set_image(outputDirectory + "resultgif.gif")
                    embed.set_thumbnail(None)
                    await activeAnimRequest.proxy.edit(embed)
                else:
                    print("Anim Complete, Gif too big.")
                    embed = hikari.Embed(title=(
                        "Animation Complete. (Gif file too large for upload)"), colour=hikari.Colour(0xFFFFFF))
                    await activeAnimRequest.proxy.edit(embed)
                activeAnimRequest = None
                awaitingFrame = None
                botBusy = False
                animationFrames = []
                return
        return
    ThreadCompletionSpeed = 2
    if awaitingEmbed != None and awaitingProxy != None:
        emb = awaitingEmbed
        prx = awaitingProxy
        ath = awaitingAuthor
        awaitingEmbed = None
        awaitingProxy = None
        awaitingAuthor = None
        rows = await generate_rows(bot)
        message = await prx.edit(emb, components=rows)
        asyncio.create_task(firehandleresponses(bot, ath, message))
        botBusy = False


async def firehandleresponses(bot, ath, message):
    await handle_responses(bot, ath, message)


# ----------------------------------
# Image to Command
# ----------------------------------
@bot.command
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.option("image_link", "image link or message ID", required=False, type=str)
@lightbulb.command("imagetocommand", "parses metadata to a command to send to get the same image")
@lightbulb.implements(lightbulb.SlashCommand)
async def imagetocommand(ctx: lightbulb.SlashContext) -> None:
    try:
        if (ctx.options.image != None):
            datas = await hikari.Attachment.read(ctx.options.image)
            mdataimage = Image.open(BytesIO(datas)).convert("RGB")
            mdataimage = mdataimage.resize((512, 512))
            url = ctx.options.image.url
        elif (ctx.options.image_link != None):
            if (is_url(ctx.options.image_link)):
                response = requests.get(ctx.options.image_link)
                url = ctx.options.image_link
                mdataimage = Image.open(
                    BytesIO(response.content)).convert("RGB")
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id, ctx.options.image_link)
                datas = await hikari.Attachment.read(messageIdResponse.embeds[0].image)
                mdataimage = Image.open(BytesIO(datas)).convert("RGB")
                mdataimage = mdataimage.resize((512, 512))
                url = messageIdResponse.embeds[0].image.url
        mdataimage = mdataimage.resize((512, 512))
        embed = hikari.Embed(title=(url.rsplit(
            '/', 1)[-1]), colour=hikari.Colour(0x56aaf8)).set_thumbnail(url)
        responseStr = "`/generate "
        if (str(mdataimage.info.get("Prompt")) != "None"):
            responseStr = responseStr+"prompt: " + \
                mdataimage.info.get("Prompt")+" "
        if (str(mdataimage.info.get("Negative Prompt")) != "None"):
            responseStr = responseStr+"negativeprompt: " + \
                mdataimage.info.get("Negative Prompt")+" "
        if (str(mdataimage.info.get("Guidance Scale")) != "None"):
            responseStr = responseStr+"guidescale: " + \
                mdataimage.info.get("Guidance Scale")+" "
        if (str(mdataimage.info.get("Inference Steps")) != "None"):
            responseStr = responseStr+"steps: " + \
                mdataimage.info.get("Inference Steps")+" "
        if (str(mdataimage.info.get("Seed")) != "None"):
            responseStr = responseStr+"seed: " + \
                mdataimage.info.get("Seed")+" "
        if (str(mdataimage.info.get("Width")) != "None"):
            responseStr = responseStr+"width: " + \
                mdataimage.info.get("Width")+" "
        if (str(mdataimage.info.get("Height")) != "None"):
            responseStr = responseStr+"height: " + \
                mdataimage.info.get("Height")+" "
        if (str(mdataimage.info.get("Scheduler")) != "None"):
            responseStr = responseStr+"sampler: " + \
                mdataimage.info.get("Scheduler")+" "
        if (str(mdataimage.info.get("Img2Img Strength")) != "None"):
            embed = hikari.Embed(
                title="This image was generated from an image input <:scootcry:1033114138366443600>", colour=hikari.Colour(0xFF0000))
            await ctx.respond(embed)
            return
        embed.description = responseStr + "`"
        rows = await generate_rows(ctx.bot)
        response = await ctx.respond(embed, components=rows)
        message = await response.message()
        await handle_responses(ctx.bot, ctx.author, message, autodelete=True)
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(
            title="Sorry, something went wrong! <:scootcry:1033114138366443600>", colour=hikari.Colour(0xFF0000))
        if (not await ctx.edit_last_response(embed)):
            await ctx.respond(embed)
        return


async def respond_with_autodelete(text: str, ctx: lightbulb.SlashContext, color=0xff0015):
    embed = hikari.Embed(title=text, colour=hikari.Colour(color))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=True)


async def processRequest(ctx: lightbulb.SlashContext, regenerate: bool, overProcess: bool = False):
    '''process ctx into a image request'''
    global curmodel
    global titles
    global outputDirectory
    global botBusy
    global previous_request
    if botBusy:
        await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
        return
    if curmodel == None or curmodel == "":
        await respond_with_autodelete("Please load a model with /changemodel", ctx)
        return
    botBusy = True
    outputDirectory = "./results/"
    try:
        if (ctx.options.image != None):
            url = ctx.options.image.url
        elif (ctx.options.image_link != None):
            if (is_url(ctx.options.image_link)):
                url = ctx.options.image_link
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id, ctx.options.image_link)
                url = messageIdResponse.embeds[0].image.url
        else:
            url = "0"

        if (ctx.options.inpaint_mask != None):
            inpainturl = ctx.options.inpaint_mask.url
        else:
            inpainturl = "0"

        # --Embed
        try:
            embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(0x56aaf8)).set_thumbnail(
                "https://i.imgur.com/21reOYm.gif").set_footer(text="", icon=curmodel["ModelThumbnail"]).set_image("https://i.imgur.com/ZCalIbz.gif")
        except:
            embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(0x56aaf8)).set_thumbnail(
                "https://i.imgur.com/21reOYm.gif").set_image("https://i.imgur.com/ZCalIbz.gif")

        respProxy = await ctx.respond(embed)
        # -------
        threadmanager = threadManager()
        userconfig = load_user_config(str(ctx.author.id))
        load_config()
        requestObject = imageRequest(ctx.options.prompt, ctx.options.negative_prompt, ctx.options.steps, ctx.options.seed, ctx.options.guidance_scale, url, ctx.options.strength, ctx.options.width,
                                     ctx.options.height, respProxy, scheduler=ctx.options.sampler, userconfig=userconfig, author=ctx.author, InpaintUrl=inpainturl, regenerate=regenerate, overProcess=overProcess)
        thread = threadmanager.New_Thread(requestObject, previous_request)
        thread.start()
    except Exception:
        traceback.print_exc()
        await respond_with_autodelete("Sorry, something went wrong! <:scootcry:1033114138366443600>", ctx)
        botBusy = False
        return


# ----------------------------------
# Generate Command
# ----------------------------------
@bot.command
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, default="DPM++", choices=["DPM++", "PNDM", "KLMS", "Euler"])
@lightbulb.option("inpaint_mask", "(Optional) mask to block off for image inpainting (white = replace, black = dont touch)", required=False, type=hikari.Attachment)
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)", required=False, type=float)
@lightbulb.option("image_link", "(Optional) image link or message ID", required=False, type=str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required=False, type=hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:15)", required=False, default=15, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidance_scale", "(Optional) Guidance scale for diffusion (Default:7)", required=False, type=float, default=7, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional)Prompt for diffusion to avoid.", required=False, default="0")
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ", required=True, default="0")
@lightbulb.command("generate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    await processRequest(ctx, False)


# ----------------------------------
# Overgenerate Command
# ----------------------------------
@bot.command
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, default="DPM++", choices=["DPM++", "PNDM", "KLMS", "Euler"])
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)", required=False, type=float)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:15)", required=False, default=15, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidance_scale", "(Optional) Guidance scale for diffusion (Default:7)", required=False, type=float, default=7, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional)Prompt for diffusion to avoid.", required=False, default="0")
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ", required=True, default="0")
@lightbulb.command("overgenerate", "runs diffusion on the result of the previous diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def overgenerate(ctx: lightbulb.SlashContext) -> None:
    await processRequest(ctx, False, True)


# ----------------------------------
# ReGenerate Command
# ----------------------------------
@bot.command
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, default="DPM++", choices=["DPM++", "PNDM", "KLMS", "Euler"])
@lightbulb.option("inpaint_mask", "(Optional) mask to block off for image inpainting (white = replace, black = dont touch)", required=False, type=hikari.Attachment)
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)", required=False, type=float)
@lightbulb.option("image_link", "(Optional) image link or message ID", required=False, type=str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required=False, type=hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:15)", required=False, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required=False, type=float, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional) Prompt for diffusion to avoid.", required=False)
@lightbulb.option("prompt", "(Optional) A detailed description of desired output, or booru tags, separated by commas. ", required=False)
@lightbulb.command("regenerate", "Regenerates diffusion from last input and optionally any changed inputs.")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    await processRequest(ctx, True)


# ----------------------------------
# Help Command
# ----------------------------------
@bot.command
@lightbulb.command("help", "get help and command info")
@lightbulb.implements(lightbulb.SlashCommand)
async def help(ctx: lightbulb.SlashContext) -> None:
    embedtext1 = (
        "**~~                   ~~ Generation ~~                   ~~**"
        "\n> **/generate**: Generates a image from a detailed description, or booru tags separated by commas"
        "\n> **/regenerate**: Re-generates last entered prompt"
        "\n> **/overgenerate**: Diffuses from last diffusion result"
        "\n**~~                      ~~ Settings ~~                         ~~**"
        "\n> **/changemodel**: switches model between stable diffusion v1.5, waifu diffusion v1.3, and yabai diffusion v???"
        "\n> **/settings**: displays a list of settings and optionally change them"
        "\n**~~                        ~~ Other ~~                        ~~**"
        "\n> **/styles**: displays a list of loaded textual inversions"
        "\n> **/styleinfo**: displays the training images of a TI"
        "\n> **/ping**: Checks connection"
        "\n**~~                          ~~ Tips ~~                          ~~**"
        "\n> More prompts (separated by commas) often result in better images, especially composition prompts."
        "\n> You can multiply prompt focus with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**"
        "\n> You can reduce focus in line like negative prompts with square brackets eg: **[**1girl**]** or **[**1girl:1.3**]**  **Default: 1.1** "
        "\n> Prompt focus modifiers can be escaped with a **\\\\** eg: **\\\\**(1girl**\\\\**), would be input as (1girl) and not be focused "
        "\n> __[Nui's Waifu Bible](https://docs.google.com/spreadsheets/d/1qBc5o6-7TIF_amqaEQhK2cNs0yfNkt4260sgh0Tgg50/)__"
        "\n> __[Composition Tags](https://danbooru.donmai.us/wiki_pages/tag_group:image_composition)__"
        "\n> __[Tag Groups](https://danbooru.donmai.us/wiki_pages/tag_groups)__"
        "\n> __[Waifu Diffusion 1.3 Release Notes](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)__"
    )
    await ctx.respond(embedtext1)


# ----------------------------------
# Admin Generate Gif Command
# ----------------------------------
@bot.command
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("animstep", "step value", required=True, type=float)
@lightbulb.option("end", "end value", required=True, type=float)
@lightbulb.option("start", "start value", required=True, type=float)
@lightbulb.option("key", "which key (guidescale, steps, strength)", required=True, type=str, choices=["guidescale", "steps", "strength"])
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, default="DPM++", choices=["DPM++", "PNDM", "KLMS", "Euler"])
@lightbulb.option("input_gif", "(Optional) gif input", required=False, type=hikari.Attachment)
@lightbulb.option("inpaint_mask", "(Optional) mask to block off for image inpainting (white = replace, black = dont touch)", required=False, type=hikari.Attachment)
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)", required=False, type=float)
@lightbulb.option("image_link", "(Optional) image link or message ID", required=False, type=str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required=False, type=hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:15)", required=False, default=15, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidance_scale", "(Optional) Guidance scale for diffusion (Default:7)", required=False, type=float, default=7, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional)Prompt for diffusion to avoid.", required=False)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ", required=True, default="0")
@lightbulb.command("admingenerategif", "Generate a series of results")
@lightbulb.implements(lightbulb.SlashCommand)
async def admingenerategif(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    global animationFrames
    global curmodel
    global titles
    global outputDirectory
    global previous_request
    animationFrames = []
    if curmodel == None or curmodel == "":
        await respond_with_autodelete("Please load a model with /changemodel", ctx)
        return
    if botBusy:
        await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
        return
    botBusy = True
    try:
        embed = hikari.Embed(title=("Animation in progress, This may take a while..."), colour=hikari.Colour(
            0xFFFFFF)).set_thumbnail("https://i.imgur.com/21reOYm.gif")
        respProxy = await ctx.respond(embed)
    except:
        pass

    botBusy = True
    outputDirectory = "./animation/"
    try:
        if (ctx.options.image != None):
            url = ctx.options.image.url
        elif (ctx.options.image_link != None):
            if (is_url(ctx.options.image_link)):
                url = ctx.options.image_link
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id, ctx.options.image_link)
                url = messageIdResponse.embeds[0].image.url
        else:
            url = "0"
        if (ctx.options.inpaint_mask != None):
            inpainturl = ctx.options.inpaint_mask.url
        else:
            inpainturl = "0"
        userconfig = load_user_config(str(ctx.author.id))
        load_config()
        global activeAnimRequest
        activeAnimRequest = animationRequest(ctx.options.prompt, ctx.options.negative_prompt, ctx.options.steps, ctx.options.seed, ctx.options.guidance_scale, url, ctx.options.strength, ctx.options.width, ctx.options.height, respProxy, scheduler=ctx.options.sampler,
                                             userconfig=userconfig, author=ctx.author, InpaintUrl=inpainturl, regenerate=False, overProcess=False, startframe=ctx.options.start, endframe=ctx.options.end, animkey=ctx.options.key, animstep=ctx.options.animstep, ingif=ctx.options.input_gif)
        global startbool
        startbool = True
    except Exception:
        traceback.print_exc()
        await respond_with_autodelete("Sorry, something went wrong! <:scootcry:1033114138366443600>", ctx)
        botBusy = False
        return
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(
            title="Sorry, something went wrong! <:scootcry:1033114138366443600>", colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        botBusy = False
        return


# ----------------------------------
# Change Model Command
# ----------------------------------
@bot.command()
@lightbulb.option("model", "which model to load", choices=model_list.keys(), required=True)
@lightbulb.command("changemodel", "changes the loaded model")
@lightbulb.implements(lightbulb.SlashCommand)
async def changemodel(ctx: lightbulb.SlashContext) -> None:
    global pipe
    global botBusy
    global model_list
    load_config()
    if not config["AllowNonAdminChangeModel"]:
        if not str(ctx.author.id) in get_admin_list():
            await respond_with_autodelete("Sorry, you must be marked as admin to change the model!", ctx)
            return
    if botBusy:
        await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
        return
    botBusy = True
    embed = hikari.Embed(
        title="Loading " + model_list[ctx.options.model]["ModelDetailedName"], colour=hikari.Colour(0xff1100))
    await ctx.respond(embed)
    change_pipeline(ctx.options.model)
    embed.color = hikari.Colour(0x00ff1a)
    embed.title = "Loaded " + \
        model_list[ctx.options.model]["ModelDetailedName"]
    await ctx.edit_last_response(embed)
    botBusy = False


# ----------------------------------
# Todo Command
# ----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("string", "string to write to todo", required=False)
@lightbulb.command("todo", "read or write the todo list")
@lightbulb.implements(lightbulb.SlashCommand)
async def todo(ctx: lightbulb.SlashContext) -> None:
    global config
    if ctx.options.string != None:
        config["TodoString"] = ctx.options.string
        save_config()
    load_config()
    embed = hikari.Embed(title="Todo:", colour=hikari.Colour(
        0xabaeff), description=config["TodoString"].replace(", ", ",\n"))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=False)


# ----------------------------------
# Settings Command
# ----------------------------------
@bot.command()
@lightbulb.option("value", "(optional if no setting) value to change it to", required=False, type=str)
@lightbulb.option("setting", "(optional) which setting to change", required=False, choices=["UseDefaultQualityPrompt", "DefaultQualityPrompt", "UseDefaultNegativePrompt", "DefaultNegativePrompt", ], type=str)
@lightbulb.command("settings", "View or modify your personal user settings")
@lightbulb.implements(lightbulb.SlashCommand)
async def settings(ctx: lightbulb.SlashContext) -> None:
    userconfig = load_user_config(str(ctx.author.id))
    if ctx.options.setting != None:
        # Bool settings
        if ctx.options.setting in ["UseDefaultNegativePrompt", "UseDefaultQualityPrompt"]:
            userconfig[ctx.options.setting] = string_to_bool(ctx.options.value)
        elif ctx.options.setting in ["DefaultNegativePrompt", "DefaultQualityPrompt"]:
            userconfig[ctx.options.setting] = ctx.options.value
        else:
            return
        save_user_config(str(ctx.author.id), userconfig)
    embed = hikari.Embed(title="User Settings:", colour=ctx.author.accent_color).set_author(
        name=ctx.member.nickname, icon=ctx.member.avatar_url)
    for key, value in userconfig.items():
        embed.add_field(str(key), str(value))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=True)


# ----------------------------------
# Admin Settings Command
# ----------------------------------
@bot.command()
@lightbulb.option("value", "(optional if no key) value to change it to", required=False, type=str)
@lightbulb.option("setting", "(optional) which setting to change", required=False, choices=["EnableNsfwFilter", "NsfwMessage", "ShowDefaultPrompts", "NewUserNegativePrompt", "NewUserQualityPrompt", "AdminList", "TodoString", "MaxSteps", "AllowNonAdminChangeModel", "AutoLoadedModel"], type=str)
@lightbulb.command("adminsettings", "View or modify settings")
@lightbulb.implements(lightbulb.SlashCommand)
async def adminsettings(ctx: lightbulb.SlashContext) -> None:
    global config
    load_config()
    if ctx.options.setting != None and ctx.options.value != None:
        if str(ctx.author.id) in get_admin_list():
            # Bools
            if ctx.options.setting in ["ShowDefaultPrompts", "EnableNsfwFilter", "AllowNonAdminChangeModel"]:
                config[ctx.options.setting] = string_to_bool(ctx.options.value)
            # Ints
            elif ctx.options.setting in ["MaxSteps"]:
                if int(ctx.options.value) > 1:
                    if int(ctx.options.value) < 500:
                        config[ctx.options.setting] = int(ctx.options.value)
                    else:
                        config[ctx.options.setting] = 500
                else:
                    config[ctx.options.setting] = 1
            # Strings
            elif ctx.options.setting in ["NsfwMessage", "AdminList", "NewUserNegativePrompt", "NewUserQualityPrompt", "TodoString", "AutoLoadedModel"]:
                config[ctx.options.setting] = ctx.options.value
            # Invalid setting
            else:
                await respond_with_autodelete("I dont understand that setting...", ctx)
                return
        else:
            await respond_with_autodelete("You must be marked as admin to change these settings...", ctx)
            return
        save_config()
    load_config()
    embed = hikari.Embed(title="Settings:", colour=hikari.Colour(0xabaeff))
    for key, value in config.items():
        embed.add_field(str(key), str(value))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=True)


# ----------------------------------
# Styles Command
# ----------------------------------
@bot.command()
@lightbulb.command("styles", "Get a list of available embeddings.")
@lightbulb.implements(lightbulb.SlashCommand)
async def styles(ctx: lightbulb.SlashContext) -> None:
    global embedlist
    SDembedliststr = ""
    WDembedliststr = ""
    identifierlist = list(
        Path("./embeddings/sd/").rglob("**/*token_identifier*"))
    for file in identifierlist:
        fileOpened = open(str(file), "r")
        SDembedliststr = SDembedliststr + fileOpened.readline() + "\n"
        fileOpened.close()
    identifierlist = list(
        Path("./embeddings/wd/").rglob("**/*token_identifier*"))
    for file in identifierlist:
        fileOpened = open(str(file), "r")
        WDembedliststr = WDembedliststr + fileOpened.readline() + "\n"
        fileOpened.close()
    embed = hikari.Embed(title="Style list:", colour=hikari.Colour(
        0xabaeff), description="These embeddings trained via textual inversion are currently loaded, add them exactly as listed in your prompt to have an effect on the output, styles may work best at beginning of the prompt, and characters/objects after.")
    embed.add_field("Waifu Diffusion:", WDembedliststr)
    embed.add_field("Stable Diffusion:", SDembedliststr)
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message)


# ----------------------------------
# Styles Example Command
# ----------------------------------
@bot.command()
@lightbulb.option("style", "the style to look for", required=True, type=str)
@lightbulb.command("styleinfo", "Get training images of a style.")
@lightbulb.implements(lightbulb.SlashCommand)
async def styleinfo(ctx: lightbulb.SlashContext) -> None:
    # find matching embedding folder
    identifierlist = list(Path("./embeddings/").rglob("**/*token_identifier*"))
    for file in identifierlist:
        fileOpened = open(str(file), "r")
        if fileOpened.readline() == ctx.options.style:
            fileOpened.close()
            #fileOpened = open(str(file.parent) + "\\README.md","r")
            embed = hikari.Embed(
                title=ctx.options.style + " - Training Dataset:", colour=hikari.Colour(0xabaeff))
            savedpath = "./embeddings/" + str(file.parent)
            if not os.path.exists(str(file.parent) + "/imageStored.txt"):
                imagepathlist = []
                imagelimitCounter = 0
                for filename in os.listdir(Path(str(file.parent) + "/concept_images/")):
                    if imagelimitCounter < 9:
                        imagepathlist.append(
                            str(file.parent) + "/concept_images/" + str(filename))
                        imagelimitCounter = imagelimitCounter + 1
                    else:
                        break
                # print(imagepathlist)
                imagesopened = []
                for imagepath in imagepathlist:

                    imagesopened.append(crop_max_square(Image.open(imagepath)).resize(
                        (512, 512), Image.Resampling.LANCZOS))
                if len(imagesopened) >= 9:
                    clippedimagelist = []
                    count = 0
                    for image in imagesopened:
                        if count < 9:
                            clippedimagelist.append(image)
                            count = count+1
                        else:
                            break
                    resultImage = image_grid(clippedimagelist, 3, 3)
                    resultImage = resultImage.resize(
                        (512, 512), Image.Resampling.LANCZOS)
                    resultImage.save(str(file.parent)+"/conceptgrid.PNG")
                    embed.set_image(str(file.parent)+"/conceptgrid.PNG")
                elif len(imagesopened) <= 3:
                    if len(imagesopened) > 0:
                        resultImage = image_grid(
                            imagesopened, 1, len(imagesopened))
                        resultImage = resultImage.resize(
                            (512, 512), Image.Resampling.LANCZOS)
                        resultImage.save(str(file.parent)+"/conceptgrid.PNG")
                        embed.set_image(str(file.parent)+"/conceptgrid.PNG")
                elif len(imagesopened) >= 4:
                    clippedimagelist = []
                    count = 0
                    for image in imagesopened:
                        if count < 4:
                            clippedimagelist.append(image)
                            count = count+1
                        else:
                            break
                    resultImage = image_grid(clippedimagelist, 2, 2)
                    resultImage = resultImage.resize(
                        (512, 512), Image.Resampling.LANCZOS)
                    resultImage.save(str(file.parent)+"/conceptgrid.PNG")
                    embed.set_image(str(file.parent)+"/conceptgrid.PNG")
            else:
                fileOpened = open(str(file.parent) + "/imageStored.txt", "r")
                embed.set_image(fileOpened.readline())
            proxy = await ctx.respond(embed)
            proxyresponse = await proxy.message()
            fileOpened = open(str(file.parent)+"/imageStored.txt", "w")
            fileOpened.writelines(proxyresponse.embeds[0].image.url)
            return
        else:
            fileOpened.close()
    await respond_with_autodelete("Style not found...", ctx)


# ----------------------------------
# Admin update commands Command
# ----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("adminupdatecommands", "update commands")
@lightbulb.implements(lightbulb.SlashCommand)
async def todo(ctx: lightbulb.SlashContext) -> None:
    await respond_with_autodelete("Commands updated.", ctx, 0x00ff1a)
    load_config()
    await bot.sync_application_commands()


# ----------------------------------
# Quit Command
# ----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("quit", "Puts kiwi to sleep")
@lightbulb.implements(lightbulb.SlashCommand)
async def quit(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("> **I'm gonna take a little nappy now... see you later... 💤**")
    await bot.close()
    await asyncio.sleep(1)
    await quit(1)


# ----------------------------------
# Toggle Negatives Command
# ----------------------------------
def set_bool_user_setting(userid: str, setting: str, value: bool):
    '''Sets a user setting to specified bool'''
    userconfig = load_user_config(userid)
    userconfig[setting] = value
    save_user_config(userid, userconfig)


def get_bool_user_setting(userid: str, setting: str) -> bool:
    '''Gets a bool user setting'''
    userconfig = load_user_config(userid)
    return userconfig[setting]


def togprompts(positive: bool, ctx: lightbulb.SlashContext) -> hikari.Embed:
    if positive:
        setting = "UseDefaultQualityPrompt"
    else:
        setting = "UseDefaultNegativePrompt"
    get_bool_user_setting(ctx.user.id, setting)
    set_bool_user_setting(ctx.user.id, setting,
                          not get_bool_user_setting(ctx.user.id, setting))
    if positive:
        if get_bool_user_setting(ctx.user.id, setting):
            embed = hikari.Embed(
                title="Your default quality prompt is now enabled. :white_check_mark:", colour=hikari.Colour(0x09ff00))
        else:
            embed = hikari.Embed(
                title="Your default quality prompt is now disabled. :x:", colour=hikari.Colour(0xff0015))
        embed.set_footer(
            "Type \"/settings setting:DefaultQualityPrompt value:\" to change it's text")
    else:
        if get_bool_user_setting(ctx.user.id, setting):
            embed = hikari.Embed(
                title="Your default negative prompt is now enabled. :white_check_mark:", colour=hikari.Colour(0x09ff00))
        else:
            embed = hikari.Embed(
                title="Your default negative prompt is now disabled. :x:", colour=hikari.Colour(0xff0015))
        embed.set_footer(
            "Type \"/settings setting:DefaultNegativePrompt value:\" to change it's text")
    return embed

# ----------------------------------
# Toggle negative prompts command
# ----------------------------------
@bot.command
@lightbulb.command("togglenegativeprompt", "toggles your user setting for using default negative prompt")
@lightbulb.implements(lightbulb.SlashCommand)
async def togglenegativeprompts(ctx: lightbulb.SlashContext) -> None:
    embed = togprompts(False, ctx)
    rows = await generate_toggle_rows(ctx.bot, False)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, ctx, True)

# ----------------------------------
# Toggle quality prompts command
# ----------------------------------
@bot.command
@lightbulb.command("togglequalityprompt", "toggles your user setting for using default quality prompt")
@lightbulb.implements(lightbulb.SlashCommand)
async def togglenegativeprompts(ctx: lightbulb.SlashContext) -> None:
    embed = togprompts(True, ctx)
    rows = await generate_toggle_rows(ctx.bot, True)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, ctx, True)


# ----------------------------------
# Start Bot
# ----------------------------------
setup()
tasks.load(bot)
bot.run()
sys.exit()
