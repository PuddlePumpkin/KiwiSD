from calendar import c
from msilib.schema import Component
from sqlite3 import Timestamp
import time
import lightbulb
import hikari
from prompt_toolkit import prompt
import torch
import sys
import datetime
import asyncio
from threading import Thread
import os
import gc
import requests
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from torch import autocast, negative
from diffusers import StableDiffusionPipeline
from io import BytesIO
import random
import traceback
from urllib.parse import urlparse
from hikari.api import ActionRowBuilder
from lightbulb.ext import tasks
from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from huggingface_hub import hf_hub_download
import glob
from pathlib import Path
import json


#----------------------------------
#Setup
#----------------------------------

modelpaths = {
"Stable Diffusion" : "C:/Users/keira/Desktop/GITHUB/Kiwi/models/stablediffusion",
"Waifu Diffusion" : "C:/Users/keira/Desktop/GITHUB/Kiwi/models/waifudiffusion",
"Yabai Diffusion" : "C:/Users/keira/Desktop/GITHUB/Kiwi/models/naidiffusers",
}
os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/embeddings")
embedlist = list(Path(".").rglob("*.[bB][iI][nN]"))
curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
config = {}
genThread = Thread()
botBusy = False
awaitingEmbed = None
awaitingProxy = None
regentitles = ["I'll try again!... ", "Sorry if I didn't do good enough... ", "I'll try my best to do better... "]
outputDirectory = "C:/Users/keira/Desktop/GITHUB/Kiwi/results/"
titles =  ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]
os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/")

def save_config():
    os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/")
    global config
    with open("kiwiconfig.json", "w") as outfile:
        json.dump(config,outfile)

def load_config():
    global config
    os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/")
    with open('kiwiconfig.json', 'r') as openfile:
        config = json.load(openfile)
#----------------------------------
#load embeddings Function
#----------------------------------
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embedsS = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embedsS.to(dtype)

  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embedsS

#----------------------------------
#Change Pipeline Function
#----------------------------------
def change_pipeline(modelpath):
    global modelpaths
    global pipe
    tokenizer = CLIPTokenizer.from_pretrained(modelpaths[modelpath],subfolder="tokenizer", use_auth_token="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")
    text_encoder = CLIPTextModel.from_pretrained(modelpaths[modelpath], subfolder="text_encoder", use_auth_token="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", torch_dtype=torch.float16)
    for file in embedlist:
        print(str(file))
        load_learned_embed_in_clip("C:/Users/keira/Desktop/GITHUB/Kiwi/embeddings/" + str(file), text_encoder, tokenizer)
    if(modelpath=="Stable Diffusion"):
        pipe = StableDiffusionPipeline.from_pretrained(modelpaths[modelpath],custom_pipeline="lpw_stable_diffusion",use_auth_token="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",torch_dtype=torch.float16, revision="fp16", text_encoder=text_encoder, tokenizer=tokenizer).to('cuda')
    else:
        pipe = StableDiffusionPipeline.from_pretrained(modelpaths[modelpath],revision="fp16", custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16, text_encoder=text_encoder, tokenizer=tokenizer).to("cuda")
    pipe.enable_attention_slicing()
change_pipeline("Waifu Diffusion")
#----------------------------------
#Filecount Function
#----------------------------------
def filecount():
    global outputDirectory
    return len([entry for entry in os.listdir(outputDirectory) if os.path.isfile(os.path.join(outputDirectory, entry))])

#----------------------------------
#isurl Function
#----------------------------------
def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
#----------------------------------
#Image Helpers
#----------------------------------
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

#----------------------------------
#Get Embed
#----------------------------------
def get_embed(Prompt,NegativePrompt, GuideScale, InfSteps, Seed, File, ImageStrength=None, Gifmode=False):
    global curmodel
    global config
    if(config["UseDefaultNegativePrompt"]):
        if NegativePrompt != None:
            NegativePrompt = NegativePrompt.replace(", " + config["DefaultNegativePrompt"],"")
            NegativePrompt = NegativePrompt.replace(config["DefaultNegativePrompt"],"")
    if ImageStrength != None:
        footer = ("{:30s} {:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:30s} {:30s}".format("Seed: "+str(Seed), "Image Strength: "+str(ImageStrength)))
    else:
        footer = ("{:30s} {:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:30s}".format("Seed: "+str(Seed)))
    f = hikari.File(File)
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image(f)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
        embed.color = hikari.Colour(0xff8f87)
    elif curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png":
        embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.color = hikari.Colour(0x56aaf8)
    else:
        embed.title = "Yabai Diffusion v??? - Result:"
        embed.color = hikari.Colour(0xff985c)
        
    
    if ((Prompt != None) and (Prompt!= "None") and (Prompt!= "")):
        embed.add_field("Prompt:",Prompt)
    if ((NegativePrompt != None) and (NegativePrompt!= "None") and (NegativePrompt!= "")):
        embed.add_field("Negative Prompt:",NegativePrompt)
    if(Gifmode):
        embed.set_footer(None)
    return embed

async def generate_rows(bot: lightbulb.BotApp):
    rows = []
    row = bot.rest.build_action_row()
    label = "🗑️"
    row.add_button(hikari.ButtonStyle.SECONDARY,label).set_label(label).add_to_container()
    rows.append(row)
    return rows

async def handle_responses(
    bot: lightbulb.BotApp,
    author: hikari.User,
    message: hikari.Message,
) -> None:
    """Watches for events, and handles responding to them."""

    # Now we need to check if the user who ran the command interacts
    # with our buttons, we stop watching after 120 seconds (2 mins) of
    # inactivity.
    with bot.stream(hikari.InteractionCreateEvent, 120).filter(
        # Here we filter out events we don't care about.
        lambda e: (
            # A component interaction is a button interaction.
            isinstance(e.interaction, hikari.ComponentInteraction)
            # Make sure the command author hit the button.
            and e.interaction.user == author
            # Make sure the button was attached to our message.
            and e.interaction.message == message
        )
    ) as stream:
        async for event in stream:
            cid = event.interaction.custom_id
            # If we haven't responded to the interaction yet, we
            # need to create the initial response. Otherwise, we
            # need to edit the initial response.
            await bot.rest.delete_message(672892614613139471,message)
            return
    # Once were back outside the stream loop, it's been 2 minutes since
    # the last interaction and it's time now to remove the buttons from
    # the message to prevent further interaction.
    await message.edit(
        # Set components to an empty list to get rid of them.
        components=[]
    )



class imageRequest(object):
    def __init__(self,Prompt=None,NegativePrompt=None,InfSteps=None,Seed=None,GuideScale=None,ImgUrl=None,Strength=None,Width=None,Height=None,Proxy=None,Config=None,resultImage=None,regenerate=False, overProcess=False):
        self.prompt = Prompt
        self.negativePrompt = NegativePrompt
        self.infSteps = InfSteps
        self.seed = Seed
        self.guideScale = GuideScale
        self.imgUrl = ImgUrl
        self.strength = Strength
        self.width = Width
        self.height = Height
        self.proxy = Proxy
        self.config = Config
        self.resultImage = resultImage
        self.regenerate = regenerate
        self.overProcess = overProcess

previous_request = imageRequest()

class threadManager(object):
    def New_Thread(self, request:imageRequest = None, previous_request:imageRequest = None):
        return genImgThreadClass(parent=self, request=request, previous_request=previous_request)
    def on_thread_finished(self, thread, data, request, proxy):
        global awaitingProxy
        global awaitingEmbed
        global previous_request
        previous_request = request
        awaitingProxy = proxy
        awaitingEmbed = data

class genImgThreadClass(Thread):
    #----------------------------------
    #Generate Image Function
    #----------------------------------
    def __init__(self, parent=None, request:imageRequest = None, previous_request:imageRequest = None):
        self.parent = parent
        self.request = request
        self.previous_request = previous_request
        super(genImgThreadClass, self).__init__()
    def run(self):
        global outputDirectory
        global pipe

        #Handle prompt
        if(self.request.prompt==None or self.request.prompt=="0"):
            self.request.prompt = ""
            if self.request.regenerate:
                if self.previous_request.prompt != None:
                    self.request.prompt = self.previous_request.prompt
        
        #Handle negative prompt
        if(self.request.negativePrompt==None or self.request.negativePrompt=="0"):
            self.request.negativePrompt = None
            if self.request.regenerate:
                if self.previous_request.negativePrompt != None:
                    self.request.negativePrompt = self.previous_request.negativePrompt
                

        #Handle Default Negative
        if not self.request.regenerate:
            if self.request.config["UseDefaultNegativePrompt"]:
                if(self.request.negativePrompt!=None):
                    if(self.request.negativePrompt==""):
                        self.request.negativePrompt = self.request.config["DefaultNegativePrompt"]
                    else:
                        self.request.negativePrompt = self.request.negativePrompt + ", " + self.request.config["DefaultNegativePrompt"]
                else:
                    self.request.negativePrompt = self.request.config["DefaultNegativePrompt"]

        #Handle infsteps
        if(self.request.infSteps=="0" or self.request.infSteps==0 or self.request.infSteps==None):
            self.request.infSteps = 30
            if self.request.regenerate:
                if self.previous_request.infSteps != None:
                    self.request.infSteps = self.previous_request.infSteps

        #Handle guidescale
        if(self.request.guideScale=="0" or self.request.guideScale==None):
            self.request.guideScale = 7.0
            if self.request.regenerate:
                if self.previous_request.guideScale != None:
                    self.request.guideScale = self.previous_request.guideScale

        #Handle Stength
        if(self.request.strength=="0" or self.request.strength==None):
            self.request.strength = 0.25
            if self.request.regenerate:
                if self.previous_request.strength != None:
                    self.request.strength = self.previous_request.strength

        #Handle ImgUrl
        if(self.request.imgUrl=="0" or self.request.imgUrl==None): 
            self.request.imgUrl = None
            if self.request.regenerate:
                if self.previous_request.imgUrl != None:
                    self.request.imgUrl = self.previous_request.imgUrl
                else:
                    self.request.strength = None
            else:
                self.request.strength = None
                

        #Handle seed
        if(self.request.seed==0 or self.request.seed==None):
            self.request.seed = random.randint(1,100000000)
            #if self.request.regenerate:
            #    if self.previous_request.seed != None:
            #        self.request.seed = self.previous_request.seed
            generator = torch.Generator("cuda").manual_seed(self.request.seed)

        #Handle Width
        if(self.request.width==0 or self.request.width== "0" or self.request.width==None):
                self.request.width = 512
                if self.request.regenerate:
                    if self.previous_request.width != None:
                        self.request.width = self.previous_request.width

        #Handle Height
        if(self.request.height==0 or self.request.height== "0" or self.request.height==None):
                self.request.height = 512
                if self.request.regenerate:
                    if self.previous_request.height != None:
                        self.request.height = self.previous_request.height
        
        #handle overprocess
        if self.request.regenerate:
            if self.previous_request.overProcess != None:
                self.request.overProcess = self.previous_request.overProcess

        #Load Image
        if self.request.imgUrl == None:
            init_image = None
        elif(not self.request.overProcess):
            print("Loading image from url: " + self.request.imgUrl)
            response = requests.get(self.request.imgUrl)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            #Crop and resize
            init_image = crop_max_square(init_image)
            init_image = init_image.resize((512, 512),Image.Resampling.LANCZOS)
        if self.request.overProcess:
            print("Loading image from previous_request.resultImage")
            init_image = self.previous_request.resultImage

        #Set Metadata
        metadata = PngInfo()
        if ((self.request.prompt != None) and (self.request.prompt!= "None") and (self.request.prompt!= "")):
            metadata.add_text("Prompt", self.request.prompt)
        if ((self.request.negativePrompt != None) and (self.request.negativePrompt != "None") and (self.request.negativePrompt != "")):
            metadata.add_text("Negative Prompt", self.request.negativePrompt)
        metadata.add_text("Guidance Scale", str(self.request.guideScale))
        metadata.add_text("Inference Steps", str(self.request.infSteps))
        metadata.add_text("Seed", str(self.request.seed))
        metadata.add_text("Width", str(self.request.width))
        metadata.add_text("Height", str(self.request.height))

        #Generate
        with autocast("cuda"):
            def dummy_checker(images, **kwargs): return images, False
            pipe.safety_checker = dummy_checker
            if self.request.strength != None:
                image = pipe(self.request.prompt,height = self.request.height, width = self.request.width, generator = generator, init_image = init_image, negative_prompt=self.request.negativePrompt, strength=(1-self.request.strength), guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps).images[0]
                metadata.add_text("Img2Img Strength", str(self.request.strength))
            else:
                image = pipe(self.request.prompt,height = self.request.height, width = self.request.width, generator = generator, init_image = init_image, negative_prompt=self.request.negativePrompt, guidance_scale=self.request.guideScale, num_inference_steps=self.request.infSteps).images[0]
        countStr = str(filecount()+1)
        while os.path.exists(outputDirectory + str(countStr) + ".png"):
            countStr = int(countStr)+1

        #Process Result
        self.request.resultImage = image
        image.save(outputDirectory + str(countStr) + ".png", pnginfo=metadata)
        outEmbed = get_embed(self.request.prompt,self.request.negativePrompt,self.request.guideScale,self.request.infSteps,self.request.seed,outputDirectory + str(countStr) + ".png",self.request.strength)
        self.parent and self.parent.on_thread_finished(self, outEmbed, self.request, self.request.proxy)



#----------------------------------    
#Instantiate a Bot instance
#----------------------------------
bot = lightbulb.BotApp(
    token="***REMOVED***",
    prefix="-",
    default_enabled_guilds=672718048343490581,
    intents=hikari.Intents.ALL,
    help_class=None
    )
tasks.load(bot)

#----------------------------------    
#Bot Ready Event
#----------------------------------
@bot.listen(hikari.ShardReadyEvent)
async def ready_listener(_):
    await bot.rest.create_message(672892614613139471, "> I'm awake running waifu diffusion v1.3! Type **/help** for help!")


#----------------------------------
#Ping Command
#----------------------------------
@bot.command
@lightbulb.command("ping", "checks the bot is alive")
@lightbulb.implements(lightbulb.SlashCommand)
async def ping(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    await ctx.respond("Pong!")
    botBusy = False

#----------------------------------
#Metadata Command
#----------------------------------
@bot.command
@lightbulb.option("image", "input image", required = False,type = hikari.Attachment)
@lightbulb.option("imagelink", "image link or message ID", required = False, type = str)
@lightbulb.command("metadata", "check metadata of an image")
@lightbulb.implements(lightbulb.SlashCommand)
async def metadata(ctx: lightbulb.SlashContext) -> None:
    if (ctx.options.image != None):
        datas = await hikari.Attachment.read(ctx.options.image)
        mdataimage = Image.open(BytesIO(datas)).convert("RGB")
        mdataimage = mdataimage.resize((512, 512))
        url = ctx.options.image.url
    elif(ctx.options.imagelink != None):
        if(is_url(ctx.options.imagelink)):
            response = requests.get(ctx.options.imagelink)
            url = ctx.options.imagelink
            mdataimage = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id,ctx.options.imagelink)
            datas = await hikari.Attachment.read(messageIdResponse.embeds[0].image)
            mdataimage = Image.open(BytesIO(datas)).convert("RGB")
            mdataimage = mdataimage.resize((512, 512))
            url = messageIdResponse.embeds[0].image.url
        mdataimage = mdataimage.resize((512, 512))
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    embed = hikari.Embed(title=(url.rsplit('/', 1)[-1]),colour=hikari.Colour(0x56aaf8)).set_thumbnail(url)
    if(str(mdataimage.info.get("Prompt")) != "None"):
        embed.add_field("Prompt:",str(mdataimage.info.get("Prompt")))
    if(str(mdataimage.info.get("Negative Prompt")) != "None"):
        embed.add_field("Negative Prompt:",str(mdataimage.info.get("Negative Prompt")))
    if(str(mdataimage.info.get("Guidance Scale")) != "None"):
        embed.add_field("Guidance Scale:",str(mdataimage.info.get("Guidance Scale")))
    if(str(mdataimage.info.get("Inference Steps")) != "None"):
        embed.add_field("Inference Steps:",str(mdataimage.info.get("Inference Steps")))
    if(str(mdataimage.info.get("Seed")) != "None"):
        embed.add_field("Seed:",str(mdataimage.info.get("Seed")))
    if(str(mdataimage.info.get("Width")) != "None"):
        embed.add_field("Width:",str(mdataimage.info.get("Width")))
    if(str(mdataimage.info.get("Height")) != "None"):
        embed.add_field("Height:",str(mdataimage.info.get("Height")))
    if(str(mdataimage.info.get("Img2Img Strength")) != "None"):
        embed.add_field("Img2Img Strength:",str(mdataimage.info.get("Img2Img Strength")))
    await ctx.respond(embed)
    botBusy = False


@tasks.task(s=2, auto_start=True)
async def checkForCompletion():
    global awaitingProxy
    global awaitingEmbed
    global botBusy
    if awaitingEmbed!=None and awaitingProxy!=None:
        emy = awaitingEmbed
        prx = awaitingProxy
        awaitingEmbed = None
        awaitingProxy = None
        await prx.edit(emy)
        botBusy = False
        
#----------------------------------
#Image to Command
#----------------------------------
@bot.command
@lightbulb.option("image", "input image", required = False, type = hikari.Attachment)
@lightbulb.option("imagelink", "image link or message ID", required = False, type = str)
@lightbulb.command("imagetocommand", "parses metadata to a command to send to get the same image")
@lightbulb.implements(lightbulb.SlashCommand)
async def imagetocommand(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    try:
        if (ctx.options.image != None):
            datas = await hikari.Attachment.read(ctx.options.image)
            mdataimage = Image.open(BytesIO(datas)).convert("RGB")
            mdataimage = mdataimage.resize((512, 512))
            url = ctx.options.image.url
        elif(ctx.options.imagelink != None):
            if(is_url(ctx.options.imagelink)):
                response = requests.get(ctx.options.imagelink)
                url = ctx.options.imagelink
                mdataimage = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id,ctx.options.imagelink)
                datas = await hikari.Attachment.read(messageIdResponse.embeds[0].image)
                mdataimage = Image.open(BytesIO(datas)).convert("RGB")
                mdataimage = mdataimage.resize((512, 512))
                url = messageIdResponse.embeds[0].image.url
        mdataimage = mdataimage.resize((512, 512))
        embed = hikari.Embed(title=(url.rsplit('/', 1)[-1]),colour=hikari.Colour(0x56aaf8)).set_thumbnail(url)
        responseStr = "`/generate "   
        if(str(mdataimage.info.get("Prompt")) != "None"):
            responseStr = responseStr+"prompt: " + mdataimage.info.get("Prompt")+" "
        if(str(mdataimage.info.get("Negative Prompt")) != "None"):
            responseStr = responseStr+"negativeprompt: "+ mdataimage.info.get("Negative Prompt")+" "
        if(str(mdataimage.info.get("Guidance Scale")) != "None"):
            responseStr = responseStr+"guidescale: "+ mdataimage.info.get("Guidance Scale")+" "
        if(str(mdataimage.info.get("Inference Steps")) != "None"):
            responseStr = responseStr+"steps: "+ mdataimage.info.get("Inference Steps")+" "
        if(str(mdataimage.info.get("Seed")) != "None"):
            responseStr = responseStr+"seed: "+ mdataimage.info.get("Seed")+" "
        if(str(mdataimage.info.get("Width")) != "None"):
            responseStr = responseStr+"width: "+ mdataimage.info.get("Width")+" "
        if(str(mdataimage.info.get("Height")) != "None"):
            responseStr = responseStr+"height: "+ mdataimage.info.get("Height")+" "
        if(str(mdataimage.info.get("Img2Img Strength")) != "None"):
            embed = hikari.Embed(title="This image was generated from an image input <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
            await ctx.respond(embed)
            botBusy = False
            return
        embed.description = responseStr + "`"
        await ctx.respond(embed)
        botBusy = False
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        if (not await ctx.edit_last_response(embed)):
            await ctx.respond(embed)
        botBusy = False
        return

#----------------------------------
#Generate Command
#----------------------------------
@bot.command
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("height", "(Optional) height of result (Default:512)", required = False,type = int, default = 512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required = False,type = int, default = 512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float)
@lightbulb.option("imagelink", "(Optional) image link or message ID", required = False, type = str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,default = 30, type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required = False, default = 0, type = int, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float,default = 7, max_value=100, min_value=-100)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False,default ="0")
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = True,default ="0")
@lightbulb.command("generate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global titles
    global outputDirectory
    global botBusy
    global config
    global previous_request
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    outputDirectory = "C:/Users/keira/Desktop/GITHUB/Kiwi/results/"
    try:
        if(ctx.options.image != None):
            url = ctx.options.image.url
        elif(ctx.options.imagelink != None):
            if(is_url(ctx.options.imagelink)):
                url = ctx.options.imagelink
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id,ctx.options.imagelink)
                url = messageIdResponse.embeds[0].image.url
        else:
            url = "0"

        #--Embed                  
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_thumbnail("https://i.imgur.com/21reOYm.gif").set_footer(text = "", icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        respProxy = await ctx.respond(embed)
        #-------
        threadmanager = threadManager()
        load_config()
        requestObject = imageRequest(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength,ctx.options.width,ctx.options.height,respProxy,config)
        thread = threadmanager.New_Thread(requestObject,previous_request)
        thread.start()
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        botBusy = False
        return

#----------------------------------
#ReGenerate Command
#----------------------------------
@bot.command
@lightbulb.option("height", "(Optional) height of result (Default:512)", required = False,type = int,choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required = False,type = int,choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float)
@lightbulb.option("imagelink", "(Optional) image link or message ID", required = False, type = str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required = False, default = 0, type = int, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, max_value=100, min_value=-100)
@lightbulb.option("negativeprompt", "(Optional) Prompt for diffusion to avoid.",required = False)
@lightbulb.option("prompt", "(Optional) A detailed description of desired output, or booru tags, separated by commas. ",required = False)
@lightbulb.command("regenerate", "Regenerates diffusion from last input and optionally any changed inputs.")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global regentitles
    global outputDirectory
    global botBusy
    global config
    global previous_request
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    outputDirectory = "C:/Users/keira/Desktop/GITHUB/Kiwi/results/"
    try:
        if(ctx.options.image != None):
            url = ctx.options.image.url
        elif(ctx.options.imagelink != None):
            if(is_url(ctx.options.imagelink)):
                url = ctx.options.imagelink
            else:
                messageIdResponse = await ctx.app.rest.fetch_message(ctx.channel_id,ctx.options.imagelink)
                url = messageIdResponse.embeds[0].image.url
        else:
            url = "0"

        #--Embed                  
        embed = hikari.Embed(title=random.choice(regentitles),colour=hikari.Colour(0x56aaf8)).set_thumbnail("https://media.discordapp.net/stickers/976356216215334953.webp").set_footer(text = "", icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        respProxy = await ctx.respond(embed)
        #-------
        threadmanager = threadManager()
        load_config()
        requestObject = imageRequest(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength,ctx.options.width,ctx.options.height,respProxy,config,regenerate=True)
        thread = threadmanager.New_Thread(requestObject,previous_request)
        thread.start()
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        botBusy = False
        return


#----------------------------------
#Help Command
#----------------------------------
@bot.command
@lightbulb.command("help", "get help and command info")
@lightbulb.implements(lightbulb.SlashCommand)
async def help(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    embedtext1 = (
    "**~~                   ~~ Generation ~~                   ~~**"
    "\n> **/generate**: Generates a image from a detailed description, or booru tags separated by commas"
    "\n> **/regenerate**: Re-generates last entered prompt"
    "\n> **/overgenerate**: Diffuses from last diffusion result"
    "\n**~~                      ~~ Settings ~~                         ~~**"
    "\n> **/changemodel**: switches model between stable diffusion v1.5, waifu diffusion v1.3, and yabai diffusion v???"
    "\n**~~                        ~~ Other ~~                        ~~**"
    "\n> **/deletelast**: Deletes the last bot message in this channel, for deleting nsfw without admin perms."
    "\n> **/ping**: Checks connection"
    "\n**~~                          ~~ Tips ~~                          ~~**"
    "\n> More prompts (separated by commas) often result in better images, especially composition prompts."
    "\n> You can multiply prompt focus with parenthesis eg: **(**1girl**)** or **(**1girl:1.3**)** **Default: 1.1**"
    "\n> You can reduce prompt focus with square brackets **[**1girl**]** / **[**1girl:1.3**]**  **Default: 1.1**"
    "\n> Prompt focus modifiers can be escaped with a **\\\\** eg: **\\\\**(1girl**\\\\**), would be input as (1girl) and not be focused "
    "\n> __[Composition Tags](https://danbooru.donmai.us/wiki_pages/tag_group:image_composition)__"
    "\n> __[Tag Groups](https://danbooru.donmai.us/wiki_pages/tag_groups)__"
    "\n> __[Waifu Diffusion 1.3 Release Notes](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)__"
    )
    await ctx.respond(embedtext1)
    botBusy = False

#----------------------------------
#Admin Generate Gif Command
#----------------------------------
@bot.command
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("animstep", "step value", required = True,type = float)
@lightbulb.option("end", "end value", required = True,type = float)
@lightbulb.option("start", "start value", required = True,type = float)
@lightbulb.option("key", "which key (guidescale, steps, strength)", required = True,type = str, choices=["guidescale", "steps", "strength"])
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,default=0.25,type = float, max_value=1, min_value=0)
@lightbulb.option("image", "image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, default=7, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:20)", required = False,type = int, default=20, max_value=100, min_value=1)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ")
@lightbulb.command("admingenerategif", "Generate a series of results")
@lightbulb.implements(lightbulb.SlashCommand)
async def admingenerategif(ctx: lightbulb.SlashContext) -> None:
    global outputDirectory
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    outputDirectory = "C:/Users/keira/Desktop/GITHUB/Kiwi/animation/"
    try:
        embed = hikari.Embed(title=("Animation in progress, This may take a while..."),colour=hikari.Colour(0xFFFFFF)).set_thumbnail("https://i.imgur.com/21reOYm.gif")
        await ctx.respond(embed)
        if ctx.options.image != None:
            genUrl = ctx.options.image.url
        else:
            genUrl = "0"
        curstep = ctx.options.start
        infsteps = ctx.options.steps
        guidance = ctx.options.guidescale
        strength = ctx.options.strength
        stepcount = 0
        imageList = []
        if ctx.options.end > ctx.options.start:
            if ctx.options.animstep > 0:
                while curstep <= ctx.options.end:
                    stepcount=stepcount + 1
                    curstep = curstep + ctx.options.animstep
                    if ctx.options.key == "guidescale":
                        guidance = float(curstep)
                    elif ctx.options.key == "steps":
                        infsteps = int(curstep)
                    elif ctx.options.key == "strength":
                        if ((strength + float(ctx.options.animstep))<=1):
                            strength = curstep 
                        else:
                            strength = 1
                    WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt,infsteps,ctx.options.seed,guidance,genUrl,strength)
                    imageList.append(prevResultImage)
            else:
                raise Exception("step not matching")
        else:
            if ctx.options.animstep < 0:
                while curstep >= ctx.options.end:
                    stepcount=stepcount + 1
                    curstep = curstep + ctx.options.animstep
                    if ctx.options.key == "guidescale":
                        guidance = float(curstep)
                    elif ctx.options.key == "steps":
                        infsteps = int(curstep)
                    elif ctx.options.key == "strength":
                        if ((strength + float(ctx.options.animstep))>=0):
                            strength = curstep 
                        else:
                            strength = 1
                    WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt,infsteps,ctx.options.seed,guidance,genUrl,strength)
                    imageList.append(prevResultImage)
            else:
                raise Exception("step not matching: " + str(ctx.options.animstep))
        imageList[0].save(outputDirectory + "resultgif.gif",save_all=True, append_images=imageList[1:], duration=86, loop=0)
        file_name = outputDirectory + "resultgif.gif"
        file_stats = os.stat(file_name)
        if((file_stats.st_size / (1024 * 1024)) < 8):
            print("Anim Complete, sending gif.")
            await ctx.edit_last_response(get_embed(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.guidescale,ctx.options.steps,ctx.options.seed,outputDirectory + "resultgif.gif",ctx.options.strength,True))
        else:
            print("Anim Complete, Gif too big.")
            embed = hikari.Embed(title=("Animation Complete. (Gif file too large for upload)"),colour=hikari.Colour(0xFFFFFF))
            await ctx.edit_last_response(embed)
        botBusy = False
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        botBusy = False
        return

#----------------------------------
#Delete Last Command
#----------------------------------
@bot.command()
@lightbulb.command("deletelast", "delete previous message in channel.")
@lightbulb.implements(lightbulb.SlashCommand)
async def deletelast(ctx: lightbulb.SlashContext) -> None:
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    """Purge a certain amount of messages from a channel."""
    if not ctx.guild_id:
        await ctx.respond("This command can only be used in a server.")
        botBusy = False
        return

    # Fetch messages that are not older than 14 days in the channel the command is invoked in
    # Messages older than 14 days cannot be deleted by bots, so this is a necessary precaution
    messages = (
        await ctx.app.rest.fetch_messages(ctx.channel_id)
        .take_until(lambda m:  m.author.id == "1032466644070572032")
        .limit(10)
        #.take_until(lambda m: datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=14) > m.created_at)
    )
    found = False
    if messages:
        for m in messages:
            print(m.author.id)
            if str(m.author.id) == "1032466644070572032":
                print("Deleted a message!")
                await ctx.app.rest.delete_message(ctx.channel_id, m)
                found = True
                break
        if found:
            await ctx.respond(f"> I'm sorry if I was too lewd >///<!")
            #else:
            #await ctx.respond(f"Could not find any message in the array!")
            await asyncio.sleep(3)
            await ctx.delete_last_response()

    else:
        await ctx.respond("Sorry >~< I couldnt find any messages I sent recently!")
    botBusy = False

#----------------------------------
#Change Model Command
#----------------------------------
@bot.command()
@lightbulb.option("model", "which model to load",choices=["Stable Diffusion","Waifu Diffusion","Yabai Diffusion"],required=True)
@lightbulb.command("changemodel", "switches model between stable diffusion / waifu diffusion / yabai diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def changemodel(ctx: lightbulb.SlashContext) -> None:
    global pipe
    global curmodel
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    if ctx.options.model.startswith("S"):
        await ctx.respond("> **Loading Stable Diffusion v1.5**")
        change_pipeline("Stable Diffusion")
        await ctx.edit_last_response("> **Loaded Stable Diffusion v1.5**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"
    elif ctx.options.model.startswith("W"):
        await ctx.respond("> **Loading Waifu Diffusion v1.3**")
        change_pipeline('Waifu Diffusion')
        await ctx.edit_last_response("> *Loaded Waifu Diffusion v1.3**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
    elif ctx.options.model.startswith("Y"):
        await ctx.respond("> **Loading Yabai Diffusion v???**")
        change_pipeline('Yabai Diffusion')
        await ctx.edit_last_response("> **Loaded Yabai Diffusion v???**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1038241897514283109/YD-01.png"
    else:
        await ctx.respond("> **I don't understand** <:scootcry:1033114138366443600>")
    botBusy = False

#----------------------------------
#Todo Command
#----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("string", "string to write to todo",required=False)
@lightbulb.command("todo", "read or write the todo list")
@lightbulb.implements(lightbulb.SlashCommand)
async def todo(ctx: lightbulb.SlashContext) -> None:
    global config
    if ctx.options.string != None:
        config["TodoString"] = ctx.options.string
        save_config()
    load_config()
    rows = await generate_rows(ctx.bot)
    embed = hikari.Embed(title="Todo:",colour=hikari.Colour(0xabaeff),description=config["TodoString"].replace(", ",",\n"))
    response = await ctx.respond(embed,components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message)

#----------------------------------
#Toggle Negative Prompts Command
#----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("option", "Whether or not to append default negative prompts for quality",required=False,type=bool, choices=[True,False])
@lightbulb.command("togglenegativeprompts", "Enable or disable default negative prompts (Enable for quality boost)")
@lightbulb.implements(lightbulb.SlashCommand)
async def todo(ctx: lightbulb.SlashContext) -> None:
    global config
    if ctx.options.option != None:
        config["UseDefaultNegativePrompt"] = ctx.options.option
        save_config()
    else:
        load_config()
        config["UseDefaultNegativePrompt"] = not config["UseDefaultNegativePrompt"]
        save_config()
    load_config()
    embed = hikari.Embed(title="Default negative prompts set to: " + str(config["UseDefaultNegativePrompt"]),colour=hikari.Colour(0xabaeff))
    await ctx.respond(embed)

#----------------------------------
#Styles Command
#----------------------------------
@bot.command()
@lightbulb.command("styles", "Get a list of available embeddings.")
@lightbulb.implements(lightbulb.SlashCommand)
async def styles(ctx: lightbulb.SlashContext) -> None:
    global embedlist
    SDembedliststr = ""
    WDembedliststr = ""
    os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/embeddings/sd")
    identifierlist = list(Path(".").rglob("**/*token_identifier*"))
    for file in identifierlist:
        print(file)
        fileOpened = open(str(file),"r")
        SDembedliststr = SDembedliststr + fileOpened.readline() + "\n"
        fileOpened.close()
    os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/embeddings/wd")
    identifierlist = list(Path(".").rglob("**/*token_identifier*"))
    for file in identifierlist:
        print(file)
        fileOpened = open(str(file),"r")
        WDembedliststr = WDembedliststr + fileOpened.readline() + "\n"
        fileOpened.close()
    os.chdir("C:/Users/keira/Desktop/GITHUB/Kiwi/")
    embed = hikari.Embed(title="Style list:",colour=hikari.Colour(0xabaeff),description="These embeddings trained via textual inversion are currently loaded, add them exactly as listed in your prompt to have an effect on the output, styles may work best at beginning of the prompt, and characters/objects after.")
    embed.add_field("Waifu Diffusion:",WDembedliststr)
    embed.add_field("Stable Diffusion:",SDembedliststr)
    await ctx.respond(embed)


#----------------------------------
#Admin update commands Command
#----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("adminupdatecommands", "update commands")
@lightbulb.implements(lightbulb.SlashCommand)
async def todo(ctx: lightbulb.SlashContext) -> None:
    await bot.sync_application_commands()
    await ctx.respond("> Commands Updated.")

#----------------------------------
#Quit Commandl
#----------------------------------
@bot.command()
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.command("quit", "Puts kiwi to sleep")
@lightbulb.implements(lightbulb.SlashCommand)
async def quit(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("> **I'm gonna take a little nappy now... see you later... 💤**")
    await bot.close()
    await asyncio.sleep(1)
    await quit(1)

bot.run()
sys.exit()