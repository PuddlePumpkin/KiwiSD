import asyncio
from asyncio.subprocess import Process
import gc
import json
import os
import random
import shutil
import sys
import traceback
import datetime
import re
from DefocusBlur import DefocusBlur, DefocusBlur_random
import subprocess
try:
    noxformers = False
    import xformers
except:
    noxformers = True
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import wget
import hikari
import lightbulb
import requests
import torch
import send2trash
import warnings
import ffencode
from lightbulb.ext import tasks
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from PIL import ImageEnhance
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
import convertckpt
from diffusers import (DDIMScheduler, DDPMScheduler, DiffusionPipeline,
                       DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler)

os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
model_list = {}
def convert_model(ckptpath, vaepath=None, dump_path=None):
    convertckpt.convert_model(ckptpath, vaepath, dump_path=dump_path)
pipe = None
# ----------------------------------
# Configs
# ----------------------------------
def load_config():
    '''loads admin config file'''
    global config
    global loadingThumbnail
    global loadingGif
    global busyThumbnail
    if not os.path.exists(str("config/kiwiconfig.json")):
        shutil.copy2("config/kiwiconfigdefault.json", "config/kiwiconfig.json")
    with open('config/kiwiconfig.json', 'r') as openfile:
        config = json.load(openfile)
        loadingThumbnail = config["LoadingThumbnail"]
        loadingGif = config["LoadingGif"]
        busyThumbnail = config["BusyThumbnail"]
        openfile.close()


def save_config():
    '''Saves admin config file'''
    global config
    with open("config/kiwiconfig.json", "w") as outfile:
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

def populate_model_list():
    global model_list
    ckptlist = list(Path("./models/").rglob("*.[cC][kK][pP][tT]"))
    #Convert Checkpoints
    for ckpt in ckptlist:
        try:
            if not os.path.exists("./models/" + str(ckpt.stem) + "/"):
                if os.path.exists("./models/" + str(ckpt.stem) + ".vae.pt"):
                    print("\nConverting with vae: " +
                          str(ckpt).replace("\\", "/") + "\n")
                    convert_model(str(ckpt), "./models/" + str(ckpt.stem) +
                                 ".vae.pt", dump_path="models/" + str(ckpt.stem) + "/")
                else:
                    print("\nConverting model without vae: " +
                          str(ckpt).replace("\\", "/") + "\n")
                    convert_model(str(ckpt), dump_path="models/" + str(ckpt.stem) + "/")
                print("\nConversion Complete: " + str(ckpt).replace("\\", "/"))
                print("Diffusers Model Path: " +
                      "models/" + str(ckpt.stem) + "/")
        except:
            print(traceback.print_exc())
            print("Failed to convert model: " + str(ckpt))
    #Iterate model folders
    for folder in next(os.walk('./models'))[1]:
        if folder != "HuggingFaceModelDescriptors":
            json_list = list(
                Path("./models/"+folder).rglob("*[mM][oO][dD][eE][lL].[jJ][sS][oO][nN]"))
            if len(json_list) == 0:
                model_list[folder] = {"ModelCommandName": folder,"ModelPath": "./models/" + folder, "ModelDetailedName": folder}
            else:
                with open(json_list[0], 'r') as openfile:
                    open_json = json.load(openfile)
                    open_json["ModelPath"] = "./models/" + folder
                    model_list[open_json["ModelCommandName"]] = open_json
                    openfile.close()
    load_config()
    #for modelkey in config["HuggingFaceModels"]:
    #    model_list[str(modelkey)] = {"ModelCommandName":str(modelkey), "ModelPath":str(config["HuggingFaceModels"][modelkey]),"ModelDetailedName":str(modelkey)}
    if config["LoadHuggingFaceModels"] == True:
        json_list = list(Path("./models/HuggingFaceModelDescriptors/").rglob("*.[jJ][sS][oO][nN]"))
        if len(json_list) == 0:
            pass
        else:
            for modeldescriptor in json_list:
                with open(modeldescriptor, 'r') as openfile:
                    open_json = json.load(openfile)
                    open_json["ModelPath"] = open_json["HuggingFacePath"]
                    model_list[open_json["ModelCommandName"]] = open_json
                    openfile.close()
    if model_list == {}:
        sys.exit("\nKiwi does not work without a model in the models directory, see readme.md for more info.\n")
    print(model_list.keys())
populate_model_list()


# ----------------------------------
# Classes
# ----------------------------------
class postprocess(object):
    def __init__(self, ContrastOffset = 0, SaturationOffset = 0, LightnessOffset = 0, SharpnessOffset = 0):
        self.LightnessOffset = LightnessOffset
        self.SaturationOffset = SaturationOffset
        self.ContrastOffset = ContrastOffset
        self.SharpnessOffset = SharpnessOffset

class imageRequest(object):
    def __init__(self, Prompt=None, NegativePrompt=None, InfSteps=None, Seed=None, GuideScale=None, ImgUrl=None, Strength=None, Width=None, Height=None, Proxy=None, resultImage=None, regenerate=False, overProcess=False, scheduler=None, userconfig=None, author=None, InpaintUrl=None, isAnimation=False, gifFrame=None, doLabel = False, labelKey = None, fontsize = None, success = True, context:lightbulb.SlashContext = None, postprocess:postprocess = None):
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
        self.fontsize = fontsize
        self.success = success
        self.context = context
        self.postprocess = postprocess


class animationRequest(object):
    def __init__(self, Prompt=None, NegativePrompt=None, InfSteps=None, Seed=None, GuideScale=None, ImgUrl=None, Strength=None, Width=None, Height=None, Proxy=None, resultImage=None, regenerate=False, overProcess=False, scheduler=None, userconfig=None, author=None, InpaintUrl=None, isAnimation=True, startframe=None, endframe=None, animkey=None, animation_step=None, ingif=None, LabelFrames=False, fontsize = None, fps = 10, context:lightbulb.SlashContext = None, postprocess = None):
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
        self.animation_step = animation_step
        self.currentstep = startframe
        self.ingif = ingif
        self.fontsize = fontsize
        self.fps = fps
        self.context = context
        self.postprocess = postprocess


class changeModelThreadManager(object):
    def New_Thread(self, modelname,context):
        return changeModelThreadClass(parent=self,modelname=modelname,context=context)
    

    def on_thread_finished(self, thread, context:lightbulb.SlashContext):
        global AwaitingModelChangeFinished
        global AwaitingModelChangeContext
        AwaitingModelChangeFinished = True
        AwaitingModelChangeContext=context


class changeModelThreadClass(Thread):
    '''thread class for change model'''

    def __init__(self, parent=None, modelname:str = None,context:lightbulb.SlashContext = None):
        self.parent = parent
        self.modelname = modelname
        self.context=context
        super(changeModelThreadClass, self).__init__()

    def run(self):
        global pipe
        global curmodel
        global tokenizer
        global text_encoder
        global model_list
        pipe = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("\nChanging model to: " + self.modelname)
        tokenizer = CLIPTokenizer.from_pretrained(model_list[self.modelname]["ModelPath"], subfolder="tokenizer", use_auth_token=HFToken)
        text_encoder = CLIPTextModel.from_pretrained(model_list[self.modelname]["ModelPath"], subfolder="text_encoder", use_auth_token=HFToken, torch_dtype=torch.float16)
        print("\nLoading Embeds...")
        update_embed_list()
        for file in embedlist:
            load_learned_embed_in_clip(str(file), text_encoder, tokenizer)
        curmodel = model_list[self.modelname]
        try:
            del pipe
        except:
            pass
        gc.collect()
        try:
            if curmodel["RequireFP32"]:
                print("Model Requires FP32")
                pipe = DiffusionPipeline.from_pretrained(model_list[self.modelname]["ModelPath"], custom_pipeline="lpw_stable_diffusion", use_auth_token=HFToken,torch_dtype=torch.float32, text_encoder=text_encoder, tokenizer=tokenizer, device_map="auto").to('cuda')
            else:
                pipe = DiffusionPipeline.from_pretrained(model_list[self.modelname]["ModelPath"], custom_pipeline="lpw_stable_diffusion", use_auth_token=HFToken,torch_dtype=torch.float16, revision="fp16",text_encoder=text_encoder, tokenizer=tokenizer, device_map="auto").to('cuda')
        except:
            pipe = DiffusionPipeline.from_pretrained(model_list[self.modelname]["ModelPath"], custom_pipeline="lpw_stable_diffusion", use_auth_token=HFToken,torch_dtype=torch.float16, revision="fp16",text_encoder=text_encoder, tokenizer=tokenizer, device_map="auto").to('cuda')    
        del tokenizer
        del text_encoder
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        print("\n" + self.modelname + " loaded.\n")
        pipe.enable_attention_slicing()
        if not noxformers:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        self.parent and self.parent.on_thread_finished(self, self.context)

        
class threadManager(object):
    def New_Thread(self, request: imageRequest = None):
        return genImgThreadClass(parent=self, request=request)


    def on_thread_finished(self, thread, data: hikari.Embed, request: imageRequest):
        global awaitingEmbed
        global awaitingRequest
        global animationFrames
        global awaitingFrame
        if request.isAnimation:
            awaitingFrame = request.resultImage
            return
        awaitingRequest = request
        awaitingEmbed = data
        requestQueue.pop(0)
if not os.path.exists("./results"):
    os.makedirs("./results")
outputDirectory = "./results/"
# ----------------------------------
# Filecount Function
# ----------------------------------
def filecount():
    global outputDirectory
    return len([entry for entry in os.listdir(outputDirectory) if os.path.isfile(os.path.join(outputDirectory, entry))])

countStr = str(filecount()+1+config["FileCountOffset"])


class genImgThreadClass(Thread):
    '''thread class for generation'''

    def __init__(self, parent=None, request: imageRequest = None):
        self.parent = parent
        self.request = request
        super(genImgThreadClass, self).__init__()


    def run(self):
        global outputDirectory
        global pipe
        global curmodel
        global botBusy
        global countStr
        try:
            # Handle Scheduler
            if (self.request.scheduler == None or self.request.scheduler == "0"):
                self.request.scheduler = "KLMS"
            try:
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
                    scheduler = DPMSolverMultistepScheduler.from_pretrained(curmodel["ModelPath"], subfolder="scheduler", solver_order=2, predict_epsilon=True, thresholding=False,
                                                                        algorithm_type="dpmsolver++", solver_type="midpoint", denoise_final=True)  # the influence of this trick is effective for small (e.g. <=10) steps)
                elif self.request.scheduler == "DDIM":
                    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",  clip_sample=False, set_alpha_to_one=False)
                pipe.scheduler = scheduler
            except Exception:
                traceback.print_exc()
            # Handle prompt
            if (self.request.prompt == None or self.request.prompt == "0"):
                self.request.prompt = ""

            # Handle negative prompt
            if (self.request.negativePrompt == None or self.request.negativePrompt == "0"):
                self.request.negativePrompt = None

            # Handle Default Negative
            if self.request.userconfig["UseDefaultNegativePrompt"]:
                if (self.request.negativePrompt != None):
                    if (self.request.negativePrompt == ""):
                        self.request.negativePrompt = self.request.userconfig["DefaultNegativePrompt"]
                    else:
                        self.request.negativePrompt = self.request.negativePrompt + ", " + self.request.userconfig["DefaultNegativePrompt"]
                else:
                    self.request.negativePrompt = self.request.userconfig["DefaultNegativePrompt"]
            self.request.negativePrompt = self.request.negativePrompt.replace(", , ",", ")
            self.request.negativePrompt = self.request.negativePrompt.replace(", ,",", ")
            # Handle Default Quality
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

            # Handle guidescale
            if (self.request.guideScale == "0" or self.request.guideScale == None):
                self.request.guideScale = 7.0

            # Handle Stength
            if (self.request.strength == None):
                self.request.strength = 0.25

            # Handle ImgUrl
            if (self.request.imgUrl == "0" or self.request.imgUrl == None):
                self.request.imgUrl = None
                if self.request.gifFrame == None:
                    self.request.strength = None

            # Handle inpaint ImgUrl
            if (self.request.inpaintUrl == "0" or self.request.inpaintUrl == None):
                self.request.inpaintUrl = None

            # Handle Width
            if (self.request.width == 0 or self.request.width == "0" or self.request.width == None):
                self.request.width = 512

            # Handle Height
            if (self.request.height == 0 or self.request.height == "0" or self.request.height == None):
                self.request.height = 512

            # Handle seed
            if (self.request.seed == 0 or self.request.seed == None):
                self.request.seed = random.randint(1, 100000000)
            generator = torch.Generator("cuda").manual_seed(self.request.seed)

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
            if self.request.gifFrame != None:
                # initialize init image from gif frame
                init_image = self.request.gifFrame
                init_image = crop_and_resize(
                    init_image, self.request.width, self.request.height)
                init_image = init_image.resize(
                    (self.request.width, self.request.height), Image.Resampling.LANCZOS)
                init_image = init_image.convert("RGB")
            # Load inpaint Image
            if self.request.inpaintUrl != None:
                inpaint_image = None
                print("Loading inpaint mask from url: " + self.request.inpaintUrl)
                response = requests.get(self.request.inpaintUrl)
                inpaint_image = Image.open(
                    BytesIO(response.content)).convert("RGB")
                #Crop and resize
                inpaint_image = crop_and_resize(
                    inpaint_image, self.request.width, self.request.height)
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
                try:
                    metadata.add_text("Model", curmodel["ModelDetailedName"])
                except:
                    metadata.add_text("Model", curmodel["ModelCommandName"])
            except:
                metadata.add_text("Model", "Stable Diffusion v2.1")

            # Generate
            with autocast("cuda"):
                def dummy_checker(images, **kwargs): return images, False
                try:
                    if not curmodel["UsesSD2"]:
                        pipe.safety_checker = dummy_checker
                except:
                    pipe.safety_checker = dummy_checker
                if self.request.strength != None:
                    if self.request.inpaintUrl == None:
                        print("Strength = " + str(1-self.request.strength))
                        image = pipe.img2img(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, image=init_image,
                                                negative_prompt=self.request.negativePrompt, strength=(1-(self.request.strength*0.89)), guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps).images[0]
                    else:
                        image = pipe.inpaint(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, image=init_image, negative_prompt=self.request.negativePrompt, strength=(
                            1-(self.request.strength*0.89)), mask_image=inpaint_image, guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps).images[0]
                    metadata.add_text("Img2Img Strength",
                                    str(self.request.strength))
                else:
                    image = pipe.text2img(prompt=self.request.prompt, height=self.request.height, width=self.request.width, generator=generator, image=init_image,
                                            negative_prompt=self.request.negativePrompt, guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps).images[0]
            #Post Processing
            try:
                if self.request.postprocess.ContrastOffset != 0:
                    filter = ImageEnhance.Contrast(image)
                    image = filter.enhance(self.request.postprocess.ContrastOffset+1)
            except:
                pass
            try:
                if self.request.postprocess.SaturationOffset != 0:
                    filter = ImageEnhance.Color(image)
                    image = filter.enhance(self.request.postprocess.SaturationOffset+1)
            except:
                pass
            try:
                if self.request.postprocess.LightnessOffset != 0:
                    filter = ImageEnhance.Brightness(image)
                    image = filter.enhance(self.request.postprocess.LightnessOffset+1)
            except:
                pass
            try:
                if self.request.postprocess.SharpnessOffset != 0:
                    filter = ImageEnhance.Sharpness(image)
                    image = filter.enhance(self.request.postprocess.SharpnessOffset+1)
            except:
                pass
            #End Post Processing
            
            while os.path.exists(outputDirectory + str(countStr) + ".png"):
                countStr = int(countStr)+1
            
            if self.request.doLabel:
                if self.request.labelKey!=None:
                    drawobj = ImageDraw.Draw(image)
                    font = ImageFont.truetype('src/Gidole-Regular.ttf', self.request.fontsize)
                    if self.request.labelKey == "guidescale":
                        drawobj.text((10, 10), "Guidance Scale: " + str(round(self.request.guideScale,2)),font=font, fill =(255, 255, 255),stroke_width=2,stroke_fill=(0,0,0))
                    if self.request.labelKey == "steps":
                        drawobj.text((10, 10), "Inference Steps: " + str(round(self.request.infSteps,2)),font=font, fill =(255, 255, 255),stroke_width=2,stroke_fill=(0,0,0))
                    if self.request.labelKey == "strength":
                        drawobj.text((10, 10), "Img2Img Strength: " + str(round(self.request.strength,2)),font=font, fill =(255, 255, 255),stroke_width=2,stroke_fill=(0,0,0))
            # Process Result
            self.request.resultImage = image
            image.save(outputDirectory + str(countStr) + ".png", pnginfo=metadata)
            outEmbed = get_embed(self.request.prompt, self.request.negativePrompt, self.request.guideScale, self.request.infSteps, self.request.seed,
                outputDirectory + str(countStr) + ".png", self.request.strength, False, self.request.scheduler, self.request.userconfig, self.request.imgUrl,self.request.context)
            self.parent and self.parent.on_thread_finished(self, outEmbed, self.request)
        except:
            print(traceback.print_exc())
            self.request.success = False
            self.parent and self.parent.on_thread_finished(self, None, self.request)





def get_admin_list() -> list:
    global config
    return config["AdminList"].replace(", ", "").replace(" ,", "").replace(" , ", "").split(",")


def load_user_config(userid: str) -> dict:
    '''Load a user config from the specified user id'''
    global config
    if not os.path.exists(str("config/usersettings.json")):
        shutil.copy2("config/usersettingsdefault.json", "config/usersettings.json")
    with open('config/usersettings.json', 'r') as openfile:
        userconfig = json.load(openfile)
        openfile.close()
    if str(userid) in userconfig:
        return userconfig[str(userid)]
    else:
        # write a default config to the userid
        load_config()
        userconfig[str(userid)] = {
        "UseDefaultQualityPrompt": True,
        "DefaultQualityPrompt": config["NewUserQualityPrompt"],
        "UseDefaultNegativePrompt": False, 
        "DefaultNegativePrompt": config["NewUserNegativePrompt"],		
        "ContrastOffset":0,
		"SaturationOffset":0,
		"LightnessOffset":0,
		"SharpnessOffset":0
        }
        save_user_config(str(userid),userconfig[str(userid)])
        return userconfig[str(userid)]

def save_user_config(userid: str, saveconfig):
    '''Saves a user setting to the json file'''
    with open('config/usersettings.json', 'r') as openfile:
        userconfigs = json.load(openfile)
        userconfigs[str(userid)] = saveconfig
        openfile.close()
    with open("config/usersettings.json", "w") as outfile:
        json.dump(userconfigs, outfile, indent=4)
        outfile.close()


# ----------------------------------
# Helpers
# ----------------------------------


# ----------------------------------
# load embeddings Function
# ----------------------------------
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    try:
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
    except:
        pass

 




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
    return ImageOps.fit(pil_img,(width,height),Image.Resampling.LANCZOS)


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
def get_embed(Prompt, NegativePrompt: str, GuideScale, InfSteps, Seed, File, ImageStrength=None, Gifmode=False, Scheduler=None, UserConfig=None, imgurl=None, context:lightbulb.SlashContext = None):
    global curmodel
    global config
    if context != None:
        if not str(context.channel_id) in config["NsfwChannelIDs"].replace(", ",",").split(","):
            NegativePrompt = NegativePrompt.replace(config["SfwNegativePrompt"] + ", ","")
            if NegativePrompt == ", " or NegativePrompt == "," or NegativePrompt == " ,":
                NegativePrompt = ""
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
        try:
            embed.title = curmodel["ModelDetailedName"] + " - Result:"
        except:
            embed.title = curmodel["ModelCommandName"] + " - Result:"
    except:
        embed.title = "Stable Diffusion v2.1" + " - Result:"
    try:
        embed.color = hikari.Colour(int(curmodel["ModelColor"], 16))
    except:
        embed.color = hikari.Colour(0x8e38ff)
    if imgurl != None:
        embed.set_thumbnail(imgurl)
    if ((Prompt != None) and (Prompt != "None") and (Prompt != "")):
        embed.add_field("Prompt:", "`" + Prompt + "`")
    if ((NegativePrompt != None) and (NegativePrompt != "None") and (NegativePrompt != "")):
        embed.add_field("Negative Prompt:", "`" + NegativePrompt + "`")
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

async def generate_cancel_rows(bot: lightbulb.BotApp):
    rows = []
    row = bot.rest.build_action_row()
    label = "❌"
    row.add_button(hikari.ButtonStyle.SECONDARY, "cancel").set_label(label).add_to_container()
    rows.append(row)
    return rows

async def handle_responses(bot: lightbulb.BotApp, author: hikari.User, message: hikari.Message, ctx: lightbulb.SlashContext = None, autodelete: bool = False) -> None:
    """Watches for events, and handles responding to them."""
    with bot.stream(hikari.InteractionCreateEvent, 60).filter(lambda e: (isinstance(e.interaction, hikari.ComponentInteraction) and e.interaction.message == message)) as stream:
        global activeAnimRequest
        global awaitingFrame
        global botBusy
        global loadedgif
        global startbool
        global animationFrames
        async for event in stream:
            if event.interaction.user == author or event.interaction.user in get_admin_list():
                cid = event.interaction.custom_id
                if cid == "toggle":
                    embed = togprompts(False, ctx)
                elif cid == "togglequality":
                    embed = togprompts(True, ctx)
                elif cid == "delete":
                    await bot.rest.delete_message(message.channel_id, message)
                    return
                elif cid == "cancel":
                    activeAnimRequest = None
                    awaitingFrame = None
                    botBusy = False
                    loadedgif = None
                    startbool = False
                    animationFrames = []
                    gc.collect()
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
embedlist = []
animationFrames = []
genThread = Thread()
botBusy = False
startbool = False
awaitingEmbed = None
AwaitingModelChangeFinished = False
AwaitingModelChangeContext = None
requestQueue = []
loadedgif = None
awaitingRequest = None
awaitingFrame = None
activeAnimRequest = None
curmodel = ""
loadingThumbnail = ""
busyThumbnail = ""
loadingGif = ""
embedlist = list(Path("./embeddings/").rglob("*.[bB][iI][nN]"))
titles = ["I'll try to make that for you!...", "Maybe I could make that...",
          "I'll try my best!...", "This might be tricky to make..."]
regentitles = ["I'll try again!... ", "Sorry if I didn't do good enough... ",
               "I'll try my best to do better... "]
def update_embed_list():
    global embedlist
    embedlist = []
    embedlist = list(Path("./embeddings/").rglob("*.[bB][iI][nN]"))


# ----------------------------------
# Setup
# ----------------------------------
def setup():
    if config["AutoLoadedModel"] != "None":
        if config["AutoLoadedModel"] in model_list:
            changemodel(config["AutoLoadedModel"])
        else:
            print("Auto loaded model not found...")
    if not os.path.exists("src/Gidole-Regular.ttf"):
        print("Downloading Gidole Regular font credit: https://github.com/larsenwork/Gidole/")
        wget.download("https://github.com/larsenwork/Gidole/raw/master/Resources/GidoleFont/Gidole-Regular.ttf",out="src/Gidole-Regular.ttf")


# ----------------------------------
# Instantiate a Bot instance
# ----------------------------------
bottoken = ""
HFToken = ""
with open('config/kiwitoken.json', 'r') as openfile:
    tokendict = json.load(openfile)
    if tokendict["huggingFaceReadToken"] != "" and tokendict["huggingFaceReadToken"] != None:
        HFToken = tokendict["huggingFaceReadToken"]
    if tokendict["bottoken"] != "" and tokendict["bottoken"] != None:
        bottoken = tokendict["bottoken"]
    else:
        try:
            bottoken = os.environ["kiwitoken"]
        except:
            pass
    if tokendict["guildID"] != None and tokendict["guildID"] != "":
        guildId = int(tokendict["guildID"])
    else: 
        warnings.warn("Commands will not update quickly without a guild ID in config/kiwitoken.json",UserWarning)
        guildId = None
    openfile.close()
if bottoken == None or bottoken == "":
    sys.exit("\nYou need a bot token, see readme.md for usage instructions")
if guildId != None:
    try:
        bot = lightbulb.BotApp(token=bottoken,intents=hikari.Intents.ALL_UNPRIVILEGED,help_class=None,default_enabled_guilds=guildId, logs = "ERROR",force_color=True,banner = "banner")
    except:
        bot = lightbulb.BotApp(token=bottoken,intents=hikari.Intents.ALL_UNPRIVILEGED,help_class=None,default_enabled_guilds=guildId, logs = "ERROR",force_color=True)
else:
    try:    
        bot = lightbulb.BotApp(token=bottoken,intents=hikari.Intents.ALL_UNPRIVILEGED,help_class=None,logs= "ERROR",force_color=True,banner = "banner")
    except:
        bot = lightbulb.BotApp(token=bottoken,intents=hikari.Intents.ALL_UNPRIVILEGED,help_class=None,logs= "ERROR",force_color=True)
# ----------------------------------
# Bot ready event
# ----------------------------------
@bot.listen(hikari.ShardReadyEvent)
async def ready_listener(_):
    pass


# ----------------------------------
# Ping Command
# ----------------------------------
@bot.command
@lightbulb.command("ping", "checks the bot is alive")
@lightbulb.implements(lightbulb.SlashCommand)
async def ping(ctx: lightbulb.SlashContext) -> None:
    await respond_with_autodelete("Pong!", ctx, 0x00ff1a)

# ----------------------------------
# Image to Depthmap Command
# ----------------------------------
@bot.command
@lightbulb.option("model", "which model to use, (default dpt)", required=False, default="dpt-large", type=str, choices=["dpt-large", "glpn-nyu"])
@lightbulb.option("image_link", "image link", required=False, type=str)
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.command("image_to_depthmap", "Run dense prediction transformer model to get depth from an image")
@lightbulb.implements(lightbulb.SlashCommand)
async def image_to_depthmap(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    try:
        if not os.path.exists("./imageprocessing"):
            os.makedirs("./imageprocessing")
        if ctx.options.image != None:
            url = ctx.options.image.url
        elif ctx.options.image_link != None:
            url = ctx.options.image_link
        else:
            await respond_with_autodelete("Please include an image link or file", ctx)
            return
        if botBusy:
            await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
            return
        botBusy = True
        embed = hikari.Embed(title="Generating Depth...", colour=hikari.Colour(0x09ff00)).set_thumbnail(loadingThumbnail)
        await ctx.respond(embed)
        if ctx.options.model != "dpt-large":
            depthPath = Path("./src/ImageToDepthGlpn.py")
        else:
            depthPath = Path("./src/ImageToDepth.py")
        pathStr = depthPath.absolute()
        process: Process = await asyncio.create_subprocess_exec("python",pathStr,url)
        await process.wait()
        try:
            process.kill()
        except:pass
        embed.title = "Depth Result:"
        embed.set_image("./imageprocessing/depth.png")
        embed.set_thumbnail(url)
        await ctx.edit_last_response(embed)
        botBusy = False
    except:
        botBusy = False
        await respond_with_autodelete("Sorry, something went wrong...",ctx)

# ----------------------------------
# Depth of field Command
# ----------------------------------
@bot.command
@lightbulb.option("show_depth", "(optional) show the depthmap in the thumbnail for adjusting settings", required=False, default=False, type=bool)
@lightbulb.option("depth_brightness_offset", "(optional) lightness offset to apply to depthmap", required=False, default=0, type=float)
@lightbulb.option("depth_contrast_offset", "(optional) contrast offset to apply to depthmap", required=False, default=0, type=float)
@lightbulb.option("depthmap_link", "(optional) use a previously generated depth map", required=False, type=str)
@lightbulb.option("image_link", "image link", required=False, type=str)
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.option("blur_kernel_size", "(optional) Size of the blur kernel", required=False, max_value=100, min_value=1, type=int)
@lightbulb.command("filter_depth_of_field", "Use DPT model or supply depth map to apply depth blur to image")
@lightbulb.implements(lightbulb.SlashCommand)
async def filter_depth_of_field(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    try:
        if not os.path.exists("./imageprocessing"):
            os.makedirs("./imageprocessing")
        embed = hikari.Embed(title="Working on it...", colour=hikari.Colour(0x09ff00)).set_thumbnail(loadingThumbnail)
        await ctx.respond(embed)
        if ctx.options.image == None and ctx.options.image_link == None:
            await respond_with_autodelete("Please include an image link or file", ctx)
            return
        if ctx.options.image_link != None:
            url = ctx.options.image_link
        elif ctx.options.image != None:
            url = ctx.options.image.url
        if botBusy:
            await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
            return
        botBusy = True
        #await ctx.respond(embed)
        if ctx.options.depthmap_link == None:
            depthPath = Path("./src/ImageToDepth.py")
            pathStr = depthPath.absolute()
            process: Process = await asyncio.create_subprocess_exec("python",pathStr,url)
            await process.wait()
            try:
                process.kill()
            except:pass
            depthmask = Image.open("./imageprocessing/depth.png").convert("L")
        else:
            response = requests.get(ctx.options.depthmap_link)
            depthmask = Image.open(BytesIO(response.content)).convert("L")
        if ctx.options.image_link != None:
            response = requests.get(ctx.options.image_link)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            response = requests.get(ctx.options.image.url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        #apply blur
        if ctx.options.blur_kernel_size != None:
            blurredimg = DefocusBlur(image, ctx.options.blur_kernel_size).convert("RGB")
        else:
            blurredimg = DefocusBlur_random(image).convert("RGB")
        filter = ImageEnhance.Contrast(depthmask)
        depthmask = filter.enhance(ctx.options.depth_contrast_offset+1)
        filter = ImageEnhance.Brightness(depthmask)
        depthmask = filter.enhance(ctx.options.depth_brightness_offset+1)
        depthmask = crop_and_resize(depthmask,image.width,image.height)
        depthmask.save("./imageprocessing/depthmask.png")
        blurredimg.save("./imageprocessing/blurredimg.png")
        image.save("./imageprocessing/baseimg.png")
        final_composite = Image.composite(image,blurredimg,depthmask)
        final_composite.save("./imageprocessing/final_composite.png")
        embed = hikari.Embed(title="Result:", colour=hikari.Colour(0x09ff00))
        if ctx.options.show_depth:
            embed.set_thumbnail("./imageprocessing/depthmask.png")
        else:
            embed.set_thumbnail(url)
        embed.set_image("./imageprocessing/final_composite.png")
        await ctx.edit_last_response(embed)
        botBusy = False
    except Exception:
        traceback.print_exc()
        botBusy = False
        await respond_with_autodelete("Sorry, something went wrong...",ctx)

# ----------------------------------
# Depth to alpha Command
# ----------------------------------
@bot.command
@lightbulb.option("show_depth", "(optional) show the depthmap in the thumbnail for adjusting settings", required=False, default=False, type=bool)
@lightbulb.option("depth_brightness_offset", "(optional) lightness offset to apply to depthmap", required=False, default=0, type=float)
@lightbulb.option("depth_contrast_offset", "(optional) contrast offset to apply to depthmap", required=False, default=0, type=float)
@lightbulb.option("depthmap_link", "(optional) use a previously generated depth map", required=False, type=str)
@lightbulb.option("image_link", "image link", required=False, type=str)
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.command("filter_depth_to_alpha", "Use DPT model or supply depth map to cut out image by depth")
@lightbulb.implements(lightbulb.SlashCommand)
async def filter_depth_to_alpha(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    try:
        if not os.path.exists("./imageprocessing"):
            os.makedirs("./imageprocessing")
        embed = hikari.Embed(title="Working on it...", colour=hikari.Colour(0x09ff00)).set_thumbnail(loadingThumbnail)
        await ctx.respond(embed)
        if ctx.options.image == None and ctx.options.image_link == None:
            await respond_with_autodelete("Please include an image link or file", ctx)
            return
        if ctx.options.image_link != None:
            url = ctx.options.image_link
        elif ctx.options.image != None:
            url = ctx.options.image.url
        if botBusy:
            await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
            return
        botBusy = True
        #await ctx.respond(embed)
        if ctx.options.depthmap_link == None:
            depthPath = Path("./src/ImageToDepth.py")
            pathStr = depthPath.absolute()
            process: Process = await asyncio.create_subprocess_exec("python",pathStr,url)
            await process.wait()
            try:
                process.kill()
            except:pass
            depthmask = Image.open("./imageprocessing/depth.png").convert("L")
        else:
            response = requests.get(ctx.options.depthmap_link)
            depthmask = Image.open(BytesIO(response.content)).convert("L")
        if ctx.options.image_link != None:
            response = requests.get(ctx.options.image_link)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            response = requests.get(ctx.options.image.url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        filter = ImageEnhance.Contrast(depthmask)
        depthmask = filter.enhance(ctx.options.depth_contrast_offset+1)
        filter = ImageEnhance.Brightness(depthmask)
        depthmask = filter.enhance(ctx.options.depth_brightness_offset+1)
        depthmask = crop_and_resize(depthmask,image.width,image.height)
        depthmask.save("./imageprocessing/depthmask.png")
        image.save("./imageprocessing/baseimg.png")
        image.putalpha(depthmask)
        image.save("./imageprocessing/final_composite.png")
        embed = hikari.Embed(title="Result:", colour=hikari.Colour(0x09ff00))
        if ctx.options.show_depth:
            embed.set_thumbnail("./imageprocessing/depthmask.png")
        else:
            embed.set_thumbnail(url)
        embed.set_image("./imageprocessing/final_composite.png")
        await ctx.edit_last_response(embed)
        botBusy = False
    except Exception:
        traceback.print_exc()
        botBusy = False
        try:
            ctx.delete_last_response()
        except:
            pass
        await respond_with_autodelete("Sorry, something went wrong...",ctx)

# ----------------------------------
# Halt / Pause Command
# ----------------------------------
paused = False
if os.path.exists("./config/halt"):
    os.remove("./config/halt")
    paused = True
@bot.command
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("halt", "Pauses Bot Generation")
@lightbulb.implements(lightbulb.SlashCommand)
async def pg(ctx: lightbulb.SlashContext) -> None:
    global paused
    global pipe
    global curmodel
    if paused:
        paused = False
        await ctx.respond("Generation Un-Halted",flags=hikari.MessageFlag.EPHEMERAL)
    else:
        with open('./config/halt', 'w') as fp:
            pass
        await ctx.respond("Restarting kiwi with pause",flags=hikari.MessageFlag.EPHEMERAL)
        await bot.close()
        await asyncio.sleep(0.5)
        await quit(1)

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
    if (str(mdataimage.info.get("parameters")) != "None"):
        regex = re.compile(r"Prompt: (?P<prompts>.*)\nNegative prompt: (?P<negativeprompts>.*)\nSteps: (?P<steps>.*), Sampler: (?P<sampler>.*), CFG scale: (?P<cfg>.*), Seed: (?P<seed>.*), S")
        matchresult = regex.search("Prompt: " + str(mdataimage.info.get("parameters")))
        #matchresult.group('prompts') # Matches all tags under prompt
        #matchresult.group('negativeprompts') # Matches all negative tags
        if (str(matchresult.group('prompts')) != ""):
            embed.add_field("Prompt:", matchresult.group('prompts'))

        if (str(matchresult.group('negativeprompts')) != ""):
            embed.add_field("Negative Prompt:", matchresult.group('negativeprompts'))

        if (str(matchresult.group('cfg')) != ""):
            embed.add_field("Guidance Scale:", matchresult.group('cfg'))

        if (str(matchresult.group('steps')) != ""):
            embed.add_field("Steps", matchresult.group('steps'))

        if (str(matchresult.group('seed')) != ""):
            embed.add_field("Seed:", matchresult.group('seed'))

        if (str(matchresult.group('sampler')) != ""):
            embed.add_field("Sampler:", matchresult.group('sampler'))
    else: 
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
            embed.add_field("Sampler:", str(mdataimage.info.get("Scheduler")))

        if (str(mdataimage.info.get("Img2Img Strength")) != "None"):
            embed.add_field("Img2Img Strength:", str(mdataimage.info.get("Img2Img Strength")))
    #if (str(mdataimage.info.get("parameters")) != "None"):
    #    embed.add_field("WebUI Metadata:", "Prompt: " + str(mdataimage.info.get("parameters")))
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message, autodelete=False)

load_config()
# ----------------------------------
# Thread result listener
# ----------------------------------
async def saveResultGif():
    global awaitingEmbed
    global botBusy
    global activeAnimRequest
    global animationFrames
    global awaitingFrame
    countStr = 1
    while os.path.exists(outputDirectory + str(countStr) + ".gif"):
        countStr = int(countStr)+1
    file_name = outputDirectory + str(countStr) + ".gif"
    fps = activeAnimRequest.fps
    animationFrames[0].save(file_name, save_all=True,append_images=animationFrames[1:], duration=1000/fps, loop=0)
    try:
        video_file_name = ffencode.encode_video(fps)
    except:
        warnings.warn("Ffmpeg not found in path, no video generated.",UserWarning)
        video_file_name = None
    file_stats = os.stat(file_name)
    if ((file_stats.st_size / (1024 * 1024)) < 8):
        print("Anim Complete, sending gif.")
        #embed = hikari.Embed(title=("Animation Result:"),colour=hikari.Colour(0xFFFFFF)).set_image(outputDirectory + "resultgif.gif")
        embed = get_embed(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.guideScale, activeAnimRequest.infSteps, activeAnimRequest.seed,
                        file_name, activeAnimRequest.strength, True, activeAnimRequest.scheduler, activeAnimRequest.userconfig, activeAnimRequest.imgUrl)
        embed.set_footer(None)
        embed.set_image(file_name)
        embed.set_thumbnail(None)
        if video_file_name == None:
            await activeAnimRequest.proxy.edit(embed)
        else:
            await activeAnimRequest.proxy.edit(attachment=video_file_name,embed=embed)
    else:
        print("Anim Complete, gif too big.")
        embed = hikari.Embed(title=(
            "Animation Complete. (gif too large for upload)"), colour=hikari.Colour(0xFFFFFF))
        await activeAnimRequest.proxy.edit(attachment=video_file_name,embed=embed)
    activeAnimRequest = None
    awaitingFrame = None
    botBusy = False
    animationFrames = []
    gc.collect()
TimeSinceGen = 0
@tasks.task(s=0.5, auto_start=True)
async def ThreadCompletionLoop():
    global awaitingEmbed
    global botBusy
    global activeAnimRequest
    global animationFrames
    global awaitingFrame
    global startbool
    global loadedgif
    global awaitingRequest
    global AwaitingModelChangeFinished
    global AwaitingModelChangeContext
    global TimeSinceGen
    TimeSinceGen = TimeSinceGen + 0.5
    if curmodel == None or curmodel == "":
            TimeSinceGen = 0
    if TimeSinceGen > config["ModelTimeoutSeconds"]:
            await bot.close()
            await asyncio.sleep(1)
            await quit(1)
    #handle model changes
    if AwaitingModelChangeFinished:
        embed = hikari.Embed(
        title="Loading " + model_list[AwaitingModelChangeContext.options.model]["ModelDetailedName"], colour=hikari.Colour(0xff1100))
        embed.color = hikari.Colour(0x00ff1a)
        embed.title = "Loaded " + \
        model_list[AwaitingModelChangeContext.options.model]["ModelDetailedName"]
        await AwaitingModelChangeContext.edit_last_response(embed)
        for channel in config["ChangeModelChannelsToNotify"].replace(", ",",").split(","):
            if channel != str(AwaitingModelChangeContext.channel_id):
                messages = await bot.rest.fetch_messages(int(channel)).take_until(lambda m: datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1) > m.created_at).limit(1)
                try:
                    if messages[0].author.id == bot.get_me().id:
                        if messages[0].embeds[0].title.startswith("Loaded"):
                            await bot.rest.edit_message(int(channel),messages[0].id,embed)
                        else:
                            await bot.rest.create_message(int(channel),embed)
                    else:
                        await bot.rest.create_message(int(channel),embed)
                except:
                    await bot.rest.create_message(int(channel),embed)
        botBusy = False
        AwaitingModelChangeContext = None
        AwaitingModelChangeFinished = False
        return
    #handle animrequests
    if activeAnimRequest != None:
        if activeAnimRequest.ingif != None:
            if (awaitingFrame != None):
                animationFrames.append(awaitingFrame)
            if (awaitingFrame != None or startbool):
                if loadedgif == None:
                    response = requests.get(activeAnimRequest.ingif.url)
                    loadedgif = Image.open(
                        BytesIO(response.content))#.convert("RGB")
                try:
                    threadmanager = threadManager()
                    load_config()
                    loadedgif.seek(loadedgif.tell()+1)
                    requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.currentstep, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height, activeAnimRequest.proxy,
                                                 scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True, gifFrame=loadedgif, context=activeAnimRequest.context,postprocess=activeAnimRequest.postprocess)
                    thread = threadmanager.New_Thread(
                        requestObject)
                    thread.start()
                    awaitingFrame = None
                    startbool = False
                    return
                except EOFError:
                    countStr = 1
                    while os.path.exists(outputDirectory + str(countStr) + ".gif"):
                        countStr = int(countStr)+1
                    file_name = outputDirectory + str(countStr) + ".gif"
                    fps = activeAnimRequest.fps
                    animationFrames[0].save(file_name, save_all=True,append_images=animationFrames[1:], duration=1000/fps, loop=0)
                    try:
                        video_file_name = ffencode.encode_video(fps)
                    except:
                        warnings.warn("Ffmpeg not found in path, no video generated.",UserWarning)
                        video_file_name = None
                    file_stats = os.stat(file_name)
                    if ((file_stats.st_size / (1024 * 1024)) < 8):
                        print("Anim Complete, sending video.")
                        embed = get_embed(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.guideScale, activeAnimRequest.infSteps, activeAnimRequest.seed,
                                        file_name, activeAnimRequest.strength, True, activeAnimRequest.scheduler, activeAnimRequest.userconfig, activeAnimRequest.imgUrl)
                        embed.set_footer(None)
                        embed.set_image(file_name)
                        embed.set_thumbnail(None)
                        if video_file_name == None:
                            await activeAnimRequest.proxy.edit(embed)
                        else:
                            await activeAnimRequest.proxy.edit(attachment=video_file_name,embed=embed)
                    else:
                        print("Anim Complete, gif too big.")
                        embed = hikari.Embed(title=(
                            "Animation Complete. (Gif file too large for upload)"), colour=hikari.Colour(0xFFFFFF))
                        if video_file_name == None:
                            await activeAnimRequest.proxy.edit(embed)
                        else:
                            await activeAnimRequest.proxy.edit(attachment=video_file_name,embed=embed)
                    activeAnimRequest = None
                    awaitingFrame = None
                    botBusy = False
                    loadedgif = None
                    animationFrames = []
                    gc.collect()
                    return
        if (awaitingFrame != None):
            animationFrames.append(awaitingFrame)
        if (awaitingFrame != None or startbool):
            if activeAnimRequest.animation_step > 0:
                if activeAnimRequest.currentstep < activeAnimRequest.endframe:
                    activeAnimRequest.currentstep = activeAnimRequest.currentstep + activeAnimRequest.animation_step
                    activeAnimRequest.animation_step
                    threadmanager = threadManager()
                    load_config()
                    if activeAnimRequest.animkey == "guidescale":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.currentstep, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if activeAnimRequest.animkey == "steps":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, int(activeAnimRequest.currentstep), activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if activeAnimRequest.animkey == "strength":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.currentstep, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if string_to_bool(activeAnimRequest.labelframes):
                        requestObject.doLabel = True
                        requestObject.labelKey = activeAnimRequest.animkey
                        requestObject.fontsize = activeAnimRequest.fontsize
                    thread = threadmanager.New_Thread(
                        requestObject)
                    thread.start()
                    awaitingFrame = None
                    startbool = False
                else:
                    await saveResultGif()
                    return
            else:
                if activeAnimRequest.currentstep > activeAnimRequest.endframe:
                    activeAnimRequest.currentstep = activeAnimRequest.currentstep + activeAnimRequest.animation_step
                    activeAnimRequest.animation_step
                    threadmanager = threadManager()
                    load_config()
                    if activeAnimRequest.animkey == "guidescale":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.currentstep, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if activeAnimRequest.animkey == "steps":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, int(activeAnimRequest.currentstep), activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.strength, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if activeAnimRequest.animkey == "strength":
                        requestObject = imageRequest(activeAnimRequest.prompt, activeAnimRequest.negativePrompt, activeAnimRequest.infSteps, activeAnimRequest.seed, activeAnimRequest.guideScale, activeAnimRequest.imgUrl, activeAnimRequest.currentstep, activeAnimRequest.width, activeAnimRequest.height,
                                                    activeAnimRequest.proxy, scheduler=activeAnimRequest.scheduler, userconfig=activeAnimRequest.userconfig, author=activeAnimRequest.author, InpaintUrl=activeAnimRequest.inpaintUrl, regenerate=activeAnimRequest.regenerate, overProcess=activeAnimRequest.overProcess, isAnimation=True,context=activeAnimRequest.context)
                    if string_to_bool(activeAnimRequest.labelframes):
                        requestObject.doLabel = True
                        requestObject.labelKey = activeAnimRequest.animkey
                        requestObject.fontsize = activeAnimRequest.fontsize
                    thread = threadmanager.New_Thread(
                        requestObject)
                    thread.start()
                    awaitingFrame = None
                    startbool = False
                else:
                    await saveResultGif()
                    return
        return
    if awaitingRequest != None:
        if not awaitingRequest.success:
            prx = awaitingRequest.proxy
            ath = awaitingRequest.author
            awaitingEmbed = None
            awaitingRequest = None
            rows = await generate_rows(bot)
            emb = hikari.Embed(title=("Sorry, something went wrong!"), colour=hikari.Colour(0xff1100))
            message = await prx.edit(emb, components=rows)
            asyncio.create_task(firehandleresponses(bot, ath, message))
            botBusy = False
            return
    if awaitingEmbed != None and awaitingRequest != None:
        emb = awaitingEmbed
        prx = awaitingRequest.proxy
        ath = awaitingRequest.author
        awaitingEmbed = None
        awaitingRequest = None
        rows = await generate_rows(bot)
        message = await prx.edit(emb, components=rows)
        asyncio.create_task(firehandleresponses(bot, ath, message))
        botBusy = False
    #handle requestQueue
    if not botBusy:
        try:
            if requestQueue[0] != None:
                threadmanager = threadManager()
                thread = threadmanager.New_Thread(requestQueue[0])
                thread.start()
                botBusy = True
        except:
            pass

async def firehandleresponses(bot, ath, message):
    await handle_responses(bot, ath, message)


# ----------------------------------
# Image to Command
# ----------------------------------
@bot.command
@lightbulb.option("image", "input image", required=False, type=hikari.Attachment)
@lightbulb.option("image_link", "image link or message ID", required=False, type=str)
@lightbulb.command("image_to_command", "parses metadata to a command to send to get the same image")
@lightbulb.implements(lightbulb.SlashCommand)
async def image_to_command(ctx: lightbulb.SlashContext) -> None:
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
            responseStr = responseStr+"negative_prompt: " + \
                mdataimage.info.get("Negative Prompt")+" "
        if (str(mdataimage.info.get("Guidance Scale")) != "None"):
            responseStr = responseStr+"guidance_scale: " + \
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
            responseStr = responseStr+"strength: " + \
                mdataimage.info.get("Img2Img Strength")+" "
        embed.description = responseStr + "`"
        rows = await generate_rows(ctx.bot)
        response = await ctx.respond(embed, components=rows)
        message = await response.message()
        await handle_responses(ctx.bot, ctx.author, message, autodelete=False)
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(
            title="Sorry, something went wrong!", colour=hikari.Colour(0xFF0000))
        if (not await ctx.edit_last_response(embed)):
            await ctx.respond(embed)
        return


async def respond_with_autodelete(text: str, ctx: lightbulb.SlashContext, color=0xff0015):
    '''Generate an embed and respond to the context with the input text'''
    await ctx.respond(text, flags=hikari.MessageFlag.EPHEMERAL)


async def processRequest(ctx: lightbulb.SlashContext, regenerate: bool, overProcess: bool = False):
    '''process ctx into a image request'''
    if len(str(ctx.options.prompt)) > 800 or len(str(ctx.options.negative_prompt)) > 800:
        await respond_with_autodelete("Sorry, prompt lengths must be less than 800 characters!",ctx)
        return
    global curmodel
    global titles
    global outputDirectory
    global botBusy
    global requestQueue
    if curmodel == None or curmodel == "":
        await respond_with_autodelete("Please load a model with /changemodel", ctx)
        return
    if ctx.options.width == 768 and ctx.options.height == 768:
        await respond_with_autodelete("Sorry, only one dimension can be 768!", ctx)
        return
    if not os.path.exists("./results"):
        os.makedirs("./results")
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
        
        if (ctx.options.steps == None):
            try:
                steps = curmodel["DefaultSteps"]
            except Exception:
                steps = 15
        else:
            steps = ctx.options.steps
        
        if (ctx.options.sampler == None):
            try:
                sampler = curmodel["DefaultSampler"]
            except Exception:
                sampler = "DPM++"
        else:
            sampler = ctx.options.sampler

        if (ctx.options.guidance_scale == None):
            try:
                guidance_scale = curmodel["DefaultGuidanceScale"]
            except Exception:
                guidance_scale = 7
        else:
            guidance_scale = ctx.options.guidance_scale

        activeQueue = False
        try:
            if len(requestQueue)>0:
                activeQueue = True
        except:
            activeQueue = True
        if not botBusy and not activeQueue:
        # --Embed
            try:
                embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(0x56aaf8)).set_thumbnail(
                    loadingThumbnail).set_footer(text="", icon=curmodel["ModelThumbnail"]).set_image(loadingGif)
            except:
                embed = hikari.Embed(title=random.choice(titles), colour=hikari.Colour(0x56aaf8)).set_thumbnail(
                    loadingThumbnail).set_image(loadingGif)

            respProxy = await ctx.respond(embed)
        else:
            embed = hikari.Embed(title="We're waiting in line!", colour=hikari.Colour(0x56aaf8)).set_thumbnail(busyThumbnail)
            respProxy = await ctx.respond(embed)
        # -------
        userconfig = load_user_config(str(ctx.author.id))
        load_config()
        if ctx.options.negative_prompt == None:
            filteredNegativePrompt:str = ""
        else:
            filteredNegativePrompt:str = str(ctx.options.negative_prompt)
        filteredPrompt:str = str(ctx.options.prompt)
        if str(ctx.channel_id) in config["NsfwChannelIDs"].replace(", ",",").split(","):
            for tag in config["NsfwFilters"].replace(", ",",").split(","):
                filteredPrompt = filteredPrompt.replace(tag,"")
        else:
            filteredNegativePrompt = config["SfwNegativePrompt"] + ", " + filteredNegativePrompt
            for tag in config["SfwFilters"].replace(", ",",").split(","):
                filteredPrompt = filteredPrompt.replace(tag,"")
        pp = postprocess()
        if not ctx.options.ignore_postprocess:
            try:
                pp.ContrastOffset = userconfig["ContrastOffset"]
            except:
                pass
            try:
                pp.SaturationOffset = userconfig["SaturationOffset"]
            except:
                pass
            try:
                pp.LightnessOffset = userconfig["LightnessOffset"]
            except:
                pass
            try:
                pp.SharpnessOffset = userconfig["SharpnessOffset"]
            except:
                pass
        requestObject = imageRequest(filteredPrompt, filteredNegativePrompt, steps, ctx.options.seed, guidance_scale, url, ctx.options.strength, ctx.options.width,
                                     ctx.options.height, respProxy, scheduler=sampler, userconfig=userconfig, author=ctx.author, InpaintUrl=inpainturl, regenerate=regenerate, overProcess=overProcess, context=ctx,postprocess=pp)
        requestQueue.append(requestObject)
    except Exception:
        print("Error")
        await respond_with_autodelete("Sorry, something went wrong!", ctx)
        traceback.print_exc()
        return


# ----------------------------------
# Generate Command
# ----------------------------------
@bot.command
@lightbulb.option("ignore_postprocess", "(Optional) Runs generation without user post processing settings", required=False, type=bool, default=False, choices=[True,False])
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, choices=["DPM++", "PNDM", "KLMS", "Euler", "DDIM"])
@lightbulb.option("inpaint_mask", "(Optional) mask to block off for image inpainting (white = replace, black = dont touch)", required=False, type=hikari.Attachment)
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)",max_value=1,min_value=0, required=False, type=float)
@lightbulb.option("image_link", "(Optional) image link or message ID", required=False, type=str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required=False, type=hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion", required=False, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidance_scale", "(Optional) Guidance scale for diffusion", required=False, type=float, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional)Prompt for diffusion to avoid.", required=False)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ", required=True, default="0")
@lightbulb.command("generate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global paused
    global TimeSinceGen
    TimeSinceGen = 0
    if paused:
        await respond_with_autodelete("Sorry, generation is currently paused!",ctx)
        return
    await processRequest(ctx, False)


# ----------------------------------
# Help Command
# ----------------------------------
@bot.command
@lightbulb.command("help", "get help and command info")
@lightbulb.implements(lightbulb.SlashCommand)
async def help(ctx: lightbulb.SlashContext) -> None:
    embedtext1 = (
        "**~~                   ~~ Generation ~~                   ~~**"
        "\n> **/changemodel**: Loads a different model"
        "\n> **/generate**: Generates a image from a detailed description, or booru tags separated by commas"
        "\n> **/generategif**: Generates a gif given entered parameters"
        "\n**~~                      ~~ Settings ~~                         ~~**"
        "\n> **/settings**: displays a list of settings and optionally change them, post process settings: -1 to 1, 0 being no affect, range is unclamped."
        "\n> **/adminsettings**: displays a list of admin settings and optionally changes them"
        "\n**~~                        ~~ Other ~~                        ~~**"
        "\n> **/styles**: displays a list of loaded textual inversions"
        "\n> **/styleinfo**: displays the training images of a TI"
        "\n> **/ping**: Checks connection"
        "\n**~~                          ~~ Tips ~~                          ~~**"
        "\n> More prompts (separated by commas) often result in better images, especially composition prompts."
        "\n> You can multiply prompt focus with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**"
        "\n**~~                          ~~ Info ~~                          ~~**"
        "\n> __[Kiwi on Github](https://github.com/PuddlePumpkin/KiwiSD)__"
        "\n> **For models trained with booru labeled data:**"
        "\n> __[Nui's Waifu Bible](https://docs.google.com/spreadsheets/d/1qBc5o6-7TIF_amqaEQhK2cNs0yfNkt4260sgh0Tgg50/)__"
        "\n> __[Composition Tags](https://danbooru.donmai.us/wiki_pages/tag_group:image_composition)__"
        "\n> __[Tag Groups](https://danbooru.donmai.us/wiki_pages/tag_groups)__"
        "\n> __[Waifu Diffusion 1.3 Release Notes](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)__"
    )
    await ctx.respond(embedtext1,flags=hikari.MessageFlag.EPHEMERAL)


# ----------------------------------
# Admin Generate Gif Command
# ----------------------------------
@bot.command
@lightbulb.option("height", "(Optional) height of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required=False, type=int, default=512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("sampler", "(Optional) Which scheduler to use", required=False, type=str, default="DPM++", choices=["DPM++", "PNDM", "KLMS", "Euler", "DDIM"])
@lightbulb.option("input_gif", "(Optional) gif input", required=False, type=hikari.Attachment)
@lightbulb.option("inpaint_mask", "(Optional) mask to block off for image inpainting (white = replace, black = dont touch)", required=False, type=hikari.Attachment)
@lightbulb.option("strength", "(Optional) Strength of the input image or power of inpainting (Default:0.25)", required=False,max_value=1,min_value=0, type=float)
@lightbulb.option("image_link", "(Optional) image link or message ID", required=False, type=str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required=False, type=hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:15)", required=False, default=15, type=int, max_value=config["MaxSteps"], min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required=False, default=0, type=int, min_value=0)
@lightbulb.option("guidance_scale", "(Optional) Guidance scale for diffusion (Default:7)", required=False, type=float, default=7, max_value=100, min_value=-100)
@lightbulb.option("negative_prompt", "(Optional)Prompt for diffusion to avoid.", required=False)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ", required=True, default="0")
@lightbulb.option("animation_label_font_size", "font size of the marked label (Default:40)", required=False,default=40, type=float)
@lightbulb.option("animation_label", "should the key be labeled through animation", required=False,default="No", type=str, choices=["Yes","No"])
@lightbulb.option("animation_fps", "Framerate of finished animation (Default:10)", required=False,default=10, type=float)
@lightbulb.option("animation_step", "how far to step each frame", required=False, type=float)
@lightbulb.option("animation_end", "end value", required=False, type=float)
@lightbulb.option("animation_start", "start value", required=False, type=float)
@lightbulb.option("animation_key", "which key (guidescale, steps, strength)", required=False, type=str, choices=["guidescale", "steps", "strength"])
@lightbulb.command("generategif", "Generate a series of results")
@lightbulb.implements(lightbulb.SlashCommand)
async def generategif(ctx: lightbulb.SlashContext) -> None:
    global paused
    global botBusy
    global animationFrames
    global curmodel
    global titles
    global outputDirectory
    global TimeSinceGen
    TimeSinceGen = 0
    if paused:
        await respond_with_autodelete("Sorry, generation is currently paused!",ctx)
        return
    animationFrames = []
    load_config()
    if config["AllowNonAdminGenerateGif"] == False:
        if not str(ctx.author.id) in get_admin_list():
            await respond_with_autodelete("Sorry, you must be marked as admin to generate an animation", ctx)
            return
    if curmodel == None or curmodel == "":
        await respond_with_autodelete("Please load a model with /changemodel", ctx)
        return
    if botBusy:
        await respond_with_autodelete("Sorry, Kiwi is busy, please try again later!", ctx)
        return
    if ctx.options.input_gif == None:
        if ctx.options.animation_step == None or ctx.options.animation_key == None or ctx.options.animation_start == None or ctx.options.animation_end == None:
            await respond_with_autodelete("Gif generation requires either a input_gif or all the options: animation_key, animation_start, animation_end, and animation_step to be filled.",ctx)
            return
        if ctx.options.animation_step >= 0:
            if ctx.options.animation_end <= ctx.options.animation_start:
                await respond_with_autodelete("Your animation step must be a negative value to match that start and end value.",ctx)
                return
        if ctx.options.animation_step <= 0:
            if ctx.options.animation_end >= ctx.options.animation_start:
                await respond_with_autodelete("Your animation step must be a positive value to match that start and end value.",ctx)
                return
    if ctx.options.animation_key == "strength":
        if ctx.options.image == None and ctx.options.image_link == None:
            await respond_with_autodelete("To animate on strength, you need an input image for img2img",ctx)
            return
    #if ctx.options.input_gif != None:
        #try:ctx.options.strength = None
        #except:pass
    botBusy = True
    #try:
    embed = hikari.Embed(title=("Animation in progress, This might take a while..."), colour=hikari.Colour(
        0xFFFFFF)).set_thumbnail(loadingThumbnail)
    cancelrows = await generate_cancel_rows(ctx.bot)
    respProxy = await ctx.respond(embed,components=cancelrows)
    m = await respProxy.message()
    #except:
    #    pass

    botBusy = True
    if not os.path.exists("./animation"):
        os.makedirs("./animation")
    outputDirectory = "./animation/"
    try:
        if (ctx.options.image != None):
            url = ctx.options.image.url
        elif (ctx.options.image_link != None):
            if (is_url(ctx.options.image_link)):
                url = ctx.options.image_link
            else:
                try:
                    messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id, ctx.options.image_link)
                    url = messageIdResponse.embeds[0].image.url
                except:
                    await respond_with_autodelete("Invalid image link!",ctx)
                    botBusy = False
                    return
        else:
            url = "0"
        if (ctx.options.inpaint_mask != None):
            inpainturl = ctx.options.inpaint_mask.url
        else:
            inpainturl = "0"
        userconfig = load_user_config(str(ctx.author.id))
        load_config()
        global activeAnimRequest
        pp = postprocess()
        try:
            pp.ContrastOffset = userconfig["ContrastOffset"]
        except:
            pass
        try:
            pp.SaturationOffset = userconfig["SaturationOffset"]
        except:
            pass
        try:
            pp.LightnessOffset = userconfig["LightnessOffset"]
        except:
            pass
        try:
            pp.SharpnessOffset = userconfig["SharpnessOffset"]
        except:
            pass
        if ctx.options.negative_prompt == None:
            filteredNegativePrompt:str = ""
        else:
            filteredNegativePrompt:str = str(ctx.options.negative_prompt)
        filteredPrompt:str = str(ctx.options.prompt)
        if str(ctx.channel_id) in config["NsfwChannelIDs"].replace(", ",",").split(","):
            for tag in config["NsfwFilters"].replace(", ",",").split(","):
                filteredPrompt = filteredPrompt.replace(tag,"")
        else:
            filteredNegativePrompt = config["SfwNegativePrompt"] + ", " + filteredNegativePrompt
            for tag in config["SfwFilters"].replace(", ",",").split(","):
                filteredPrompt = filteredPrompt.replace(tag,"")
        activeAnimRequest = animationRequest(filteredPrompt, filteredNegativePrompt, ctx.options.steps, ctx.options.seed, ctx.options.guidance_scale, url, ctx.options.strength, ctx.options.width, ctx.options.height, respProxy, scheduler=ctx.options.sampler,
                                             userconfig=userconfig, author=ctx.author, InpaintUrl=inpainturl, regenerate=False, overProcess=False, startframe=ctx.options.animation_start, endframe=ctx.options.animation_end, animkey=ctx.options.animation_key, animation_step=ctx.options.animation_step, ingif=ctx.options.input_gif,LabelFrames=ctx.options.animation_label,fontsize=ctx.options.animation_label_font_size,fps=ctx.options.animation_fps, context=ctx, postprocess=pp)
        global startbool
        startbool = True
        lastpnglist = list(Path("./animation/").rglob("*.png"))
        send2trash.send2trash(lastpnglist)
        #os.remove(lastpnglist[0].absolute)
    except Exception:
        traceback.print_exc()
        await respond_with_autodelete("Sorry, something went wrong!", ctx)
        botBusy = False
        return
    await handle_responses(ctx.bot, ctx.author, m, autodelete=False)


# ----------------------------------
# Change Model Command
# ----------------------------------
@bot.command()
@lightbulb.option("model", "which model to load", choices=model_list.keys(), required=False)
@lightbulb.command("changemodel", "changes the loaded model")
@lightbulb.implements(lightbulb.SlashCommand)
async def changemodel(ctx: lightbulb.SlashContext) -> None:
    global pipe
    global botBusy
    global model_list
    global tokenizer
    global text_encoder
    global AwaitingModelChangeContext
    global paused
    global TimeSinceGen
    TimeSinceGen = 0
    if paused:
        await respond_with_autodelete("Sorry, generation is currently paused!",ctx)
        return
    if ctx.options.model == None:
        await respond_with_autodelete("You must specify a model",ctx)
        return
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

    try:
        del tokenizer
        del text_encoder
    except:
        pass
    threadmanager = changeModelThreadManager()
    thread = threadmanager.New_Thread(ctx.options.model,ctx)
    thread.start()


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
@lightbulb.option("setting", "(optional) which setting to change", required=False, choices=["UseDefaultQualityPrompt", "DefaultQualityPrompt", "UseDefaultNegativePrompt", "DefaultNegativePrompt", "SaturationOffset","LightnessOffset","ContrastOffset","SharpnessOffset"], type=str)
@lightbulb.command("settings", "View or modify your personal user settings")
@lightbulb.implements(lightbulb.SlashCommand)
async def settings(ctx: lightbulb.SlashContext) -> None:
    if len(str(ctx.options.value)) > 800:
        print("LIMIT")
        await respond_with_autodelete("Sorry, setting length must be less than 800 characters!",ctx)
        return
    userconfig = load_user_config(str(ctx.author.id))
    if ctx.options.setting != None:
        # Bool settings
        if ctx.options.setting in ["UseDefaultNegativePrompt", "UseDefaultQualityPrompt"]:
            userconfig[ctx.options.setting] = string_to_bool(ctx.options.value)
        # String settings
        elif ctx.options.setting in ["DefaultNegativePrompt", "DefaultQualityPrompt"]:
            userconfig[ctx.options.setting] = ctx.options.value
        # Number settings
        elif ctx.options.setting in ["SaturationOffset","LightnessOffset","ContrastOffset","SharpnessOffset"]:
            try:
                userconfig[ctx.options.setting] = float(ctx.options.value)
            except:
                await respond_with_autodelete("Sorry, something went wrong!", ctx)
                return
        else:
            await respond_with_autodelete("Sorry, something went wrong!", ctx)
            return
        save_user_config(str(ctx.author.id), userconfig)
    embed = hikari.Embed(title="User Settings:", colour=ctx.author.accent_color)
    if ctx.member.nickname is not None:
        embed.set_author(name=ctx.member.nickname, icon=ctx.member.avatar_url)
    else:
        embed.set_author(name=ctx.member.username, icon=ctx.member.avatar_url)
    for key, value in userconfig.items():
        if value != None and value != "":
            embed.add_field(str(key), str(value))
        else:
            embed.add_field(str(key), str("None"))
    await ctx.respond(embed, flags=hikari.MessageFlag.EPHEMERAL)


# ----------------------------------
# Admin Settings Command
# ----------------------------------
@bot.command()
@lightbulb.option("value", "(optional if no key) value to change it to", required=False, type=str)
@lightbulb.option("setting", "(optional) which setting to change", required=False, choices=["NsfwChannelIDs", "ShowDefaultPrompts", "NewUserNegativePrompt", "NewUserQualityPrompt", "AdminList", "TodoString", "MaxSteps", "AllowNonAdminChangeModel", "AllowNonAdminGenerateGif", "AutoLoadedModel","LoadingGif", "LoadingThumbnail", "BusyThumbnail", "NsfwFilters","SfwFilters", "ChangeModelChannelsToNotify"], type=str)
@lightbulb.command("adminsettings", "View or modify settings")
@lightbulb.implements(lightbulb.SlashCommand)
async def adminsettings(ctx: lightbulb.SlashContext) -> None:
    global config
    load_config()
    # Check if the user is an admin
    if str(ctx.author.id) not in get_admin_list():
        await respond_with_autodelete("You must be marked as admin to change these settings...", ctx)
        return
    # Check if setting and value are provided
    if ctx.options.setting is None or ctx.options.value is None:
        pass
    else:
        # Update the config with the provided setting and value
        if ctx.options.setting in ["ShowDefaultPrompts", "AllowNonAdminChangeModel", "AllowNonAdminGenerateGif"]:
            config[ctx.options.setting] = string_to_bool(ctx.options.value)
        elif ctx.options.setting in ["MaxSteps"]:
            config[ctx.options.setting] = min(max(int(ctx.options.value), 1), 500)
        elif ctx.options.setting in ["NsfwChannelIDs", "AdminList", "NewUserNegativePrompt", "NewUserQualityPrompt", "TodoString", "AutoLoadedModel", "LoadingGif", "LoadingThumbnail", "BusyThumbnail", "NsfwFilters", "SfwFilters", "ChangeModelChannelsToNotify"]:
            config[ctx.options.setting] = ctx.options.value
        else:
            await respond_with_autodelete("I don't understand that setting...", ctx)
            return
    save_config()
    # Create an embed with the current config values
    embed = hikari.Embed(title="Settings:", colour=hikari.Colour(0xabaeff))
    for key, value in config.items():
        if value == "" or value == None:
            value = "None"
        embed.add_field(key, value)
    await ctx.respond(embed, flags=hikari.MessageFlag.EPHEMERAL)



# ----------------------------------
# Styles Command
# ----------------------------------
@bot.command()
@lightbulb.command("styles", "Get a list of available embeddings.")
@lightbulb.implements(lightbulb.SlashCommand)
async def styles(ctx: lightbulb.SlashContext) -> None:
    global embedlist
    embedliststr = ""
    
    identifierlist = list(
        Path("./embeddings/").rglob("**/*token_identifier*"))
    for file in identifierlist:
        fileOpened = open(str(file), "r")
        embedliststr = embedliststr + fileOpened.readline() + "\n"
        fileOpened.close()
    embed = hikari.Embed(title="Style list:", colour=hikari.Colour(0xabaeff), description="These embeddings trained via textual inversion are currently loaded, add them exactly as listed in your prompt to have an effect on the output, styles may work best at beginning of the prompt, and characters/objects after.")
    if embedliststr == None or embedliststr == "":
        embedliststr = "No embeds loaded."
    embed.add_field("Embeds:",embedliststr)
    rows = await generate_rows(ctx.bot)
    response = await ctx.respond(embed, components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message)


# ----------------------------------
# Styleinfo Command
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
            if not os.path.exists(str(file.parent) + "/conceptgrid.png"):
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
                embed.set_image(str(file.parent) + "/conceptgrid.png")
            try:
                await ctx.respond(embed)
            except:
                await respond_with_autodelete("No concept images found for that style.",ctx)
            return
        else:
            fileOpened.close()
    await respond_with_autodelete("Style not found...", ctx)


# ----------------------------------
# Restart Command
# ----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("restart", "Force reboot kiwi")
@lightbulb.implements(lightbulb.SlashCommand)
async def restart(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("Rebooting",flags=hikari.MessageFlag.EPHEMERAL)
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
    set_bool_user_setting(ctx.user.id, setting, not get_bool_user_setting(ctx.user.id, setting))
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
    await ctx.respond(embed,flags=hikari.MessageFlag.EPHEMERAL)

# ----------------------------------
# Toggle quality prompts command
# ----------------------------------
@bot.command
@lightbulb.command("togglequalityprompt", "toggles your user setting for using default quality prompt")
@lightbulb.implements(lightbulb.SlashCommand)
async def togglenegativeprompts(ctx: lightbulb.SlashContext) -> None:
    embed = togprompts(True, ctx)
    await ctx.respond(embed,flags=hikari.MessageFlag.EPHEMERAL)


# ----------------------------------
# Start Bot
# ----------------------------------
setup()
tasks.load(bot)
bot.run()
sys.exit()
