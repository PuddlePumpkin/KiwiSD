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

#----------------------------------
#Globals
#----------------------------------
genThread = Thread()
curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',custom_pipeline="lpw_stable_diffusion",torch_dtype=torch.float16, revision="fp16").to('cuda')
pipe.enable_attention_slicing()
prevPrompt = ""
prevNegPrompt = ""
prevStrength = 0.25
prevUrl = ""
prevSeed = None
prevResultImage = None
prevGuideScale = None
prevInfSteps = None
overprocessbool = False
prevWidth = 512
prevHeight = 512
botBusy = False
outEmbed = None
regentitles = ["I'll try again!... ", "Sorry if I didn't do good enough... ", "I'll try my best to do better... "]
outputDirectory = "C:/Users/keira/Desktop/GITHUB/Kiwi/results/"
titles =  ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]

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
    if ImageStrength != None:
        footer = ("{:30s} {:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:30s} {:30s}".format("Seed: "+str(Seed), "Image Strength: "+str(ImageStrength)))
    else:
        footer = ("{:30s} {:30s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:30s}".format("Seed: "+str(Seed)))
    f = hikari.File(File)
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image(f)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
        embed.color = hikari.Colour(0xff8f87)
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.color = hikari.Colour(0x56aaf8)
    
    if ((Prompt != None) and (Prompt!= "None") and (Prompt!= "")):
        embed.add_field("Prompt:",Prompt)
    if ((NegativePrompt != None) and (NegativePrompt!= "None") and (NegativePrompt!= "")):
        embed.add_field("Negative Prompt:",prevNegPrompt)
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
awaitingEmbed = None
awaitingProxy = None
class threadManager(object):
    def New_Thread(self, Prompt=None,NegativePrompt=None,InfSteps=None,Seed=None,GuideScale=None,ImgUrl=None,Strength=None,Width=None,Height=None,Proxy=None):
        return genImgThreadClass(parent=self, Prompt=Prompt,NegativePrompt=NegativePrompt,InfSteps=InfSteps,Seed=Seed,GuideScale=GuideScale,ImgUrl=ImgUrl,Strength=Strength,Width=Width,Height=Height,Proxy=Proxy)
    def on_thread_finished(self, thread, data, proxy):
        global awaitingProxy
        global awaitingEmbed
        awaitingProxy = proxy
        awaitingEmbed = data


class genImgThreadClass(Thread):
    #----------------------------------
    #Generate Image Function
    #----------------------------------
    def __init__(self, parent=None, Prompt=None,NegativePrompt=None,InfSteps=None,Seed=None,GuideScale=None,ImgUrl=None,Strength=None,Width=None,Height=None,Proxy=None):
        self.parent = parent
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
        super(genImgThreadClass, self).__init__()
    def run(self):
        global prevPrompt
        global prevNegPrompt
        global prevInfSteps
        global prevGuideScale
        global prevSeed
        global prevUrl
        global prevStrength
        global prevResultImage
        global prevWidth
        global prevHeight
        global outputDirectory
        global pipe
        global overprocessbool
        global outEmbed

        #Handle negative prompt
        if(self.prompt!=None):
            if(self.prompt=="0"):
                prevPrompt = ""
                prompt = ""
            else:
                prevPrompt = self.prompt
                prompt = self.prompt
        else:
            prevPrompt = ""
            prompt = ""
        
        #Handle negative prompt
        if(self.negativePrompt!=None):
            if(self.negativePrompt=="0"):
                prevNegPrompt = None
                negativeprompt = None
            else:
                prevNegPrompt = self.negativePrompt
                negativeprompt = self.negativePrompt
        #Its okay to use prev neg prompt even if empty
        else:
            negativeprompt = prevNegPrompt

        #Handle infsteps
        if(self.infSteps!=None):
            if(self.infSteps=="0"):
                prevInfSteps = 30
                infsteps = 30
            else:
                prevInfSteps = self.infSteps
                infsteps = self.infSteps
        elif(prevInfSteps!=None):
            infsteps = prevInfSteps
        elif(prevInfSteps==None):
            prevInfSteps = 30
            infsteps = prevInfSteps

        #Handle guidescale
        if(self.guideScale!=None):
            if(self.guideScale=="0"):
                prevGuideScale = 7.0
                guidescale = 7.0
            else:
                prevGuideScale = self.guideScale
                guidescale = self.guideScale
        elif(prevGuideScale!=None):
            guidescale = prevGuideScale
        elif(prevGuideScale==None):
            prevGuideScale = 7
            guidescale = prevGuideScale

        #Handle Stength
        if(self.strength!=None):
            if(self.strength=="0"):
                prevStrength = 0.25
                strength = 0.25
            else:
                prevStrength = self.strength
                strength = self.strength
        elif(prevStrength!=None):
            strength = prevStrength
        elif(prevStrength==None):
            prevStrength = 0.25
            strength = prevStrength

        #Handle ImgUrl
        if(self.imgUrl!=None):
            if(self.imgUrl=="0"):
                prevUrl = None
                prevStrength = None
                strength = None
                imgurl = None
            else:
                prevUrl = self.imgUrl 
                imgurl = self.imgUrl
        elif(prevUrl!=None):
            imgurl = prevUrl
        elif(prevUrl==None):
            prevUrl = None
            strength = None
            prevStrength = None
            imgurl = prevUrl

        #Handle seed
        if(self.seed!=None):
            if(self.seed==0):
                prevSeed = random.randint(1,100000000)
                seed = prevSeed
                generator = torch.Generator("cuda").manual_seed(prevSeed)
            else:
                prevSeed = self.seed
                seed = self.seed
                generator = torch.Generator("cuda").manual_seed(self.seed)
        elif(prevSeed!=None):
            seed = prevSeed
            generator = torch.Generator("cuda").manual_seed(prevSeed)
        elif(prevSeed==None):
            prevSeed = random.randint(1,100000000)
            seed = prevSeed
            generator = torch.Generator("cuda").manual_seed(prevSeed)

        #Handle Width
        if(self.width!=None):
            if(self.width=="0"):
                prevWidth = 512
                width = 512
            else:
                prevWidth = self.width
                width = self.width
        elif(prevWidth!=None):
            width = prevWidth
        elif(prevWidth==None):
            prevWidth = 512
            width = prevWidth

        #Handle Height
        if(self.height!=None):
            if(self.height=="0"):
                prevHeight = 512
                height = 512
            else:
                prevHeight = self.height
                height = self.height
        elif(prevHeight!=None):
            height = prevHeight
        elif(prevHeight==None):
            prevHeight = 512
            height = prevHeight
        if imgurl != None:
            if(overprocessbool):
                print("Loading image from prevResultImage")
                init_image = prevResultImage
                overprocessbool=False
            else:
                print("Loading image from url: " + imgurl)
                response = requests.get(imgurl)
                init_image = Image.open(BytesIO(response.content)).convert("RGB")
                #Crop and resize
                init_image = crop_max_square(init_image)
                init_image = init_image.resize((512, 512),Image.Resampling.LANCZOS)
        else:
            init_image = None

        #Set Metadata
        metadata = PngInfo()
        if ((prompt != None) and (prompt!= "None") and (prompt!= "")):
            metadata.add_text("Prompt", prompt)
        if ((negativeprompt != None) and (negativeprompt!= "None") and (negativeprompt != "")):
            metadata.add_text("Negative Prompt", negativeprompt)
        metadata.add_text("Guidance Scale", str(guidescale))
        metadata.add_text("Inference Steps", str(infsteps))
        metadata.add_text("Seed", str(seed))
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))

        with autocast("cuda"):
            def dummy_checker(images, **kwargs): return images, False
            pipe.safety_checker = dummy_checker
            if strength != None:
                image = pipe(prompt, generator = generator, init_image = init_image, negative_prompt=negativeprompt, strength=(1-strength), guidance_scale=guidescale, num_inference_steps=infsteps).images[0]
                metadata.add_text("Img2Img Strength", str(strength))
            else:
                image = pipe(prompt,height = height,width = width, generator = generator, init_image = init_image, negative_prompt=negativeprompt, guidance_scale=guidescale, num_inference_steps=infsteps).images[0]
        countStr = str(filecount()+1)
        while os.path.exists(outputDirectory + str(countStr) + ".png"):
            countStr = int(countStr)+1

        prevResultImage = image
        image.save(outputDirectory + str(countStr) + ".png", pnginfo=metadata)
        outEmbed = get_embed(prompt,negativeprompt,guidescale,infsteps,seed,outputDirectory + str(countStr) + ".png",strength)
        self.parent and self.parent.on_thread_finished(self, outEmbed, self.proxy)
        return get_embed(prompt,negativeprompt,guidescale,infsteps,seed,outputDirectory + str(countStr) + ".png",strength)



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
@lightbulb.option("height", "(Optional) height of result (Default:512)", required = False,type = int, default = 512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("width", "(Optional) width of result (Default:512)", required = False,type = int, default = 512, choices=[128, 256, 384, 512, 640, 768])
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float)
@lightbulb.option("imagelink", "(Optional) image link or message ID", required = False, type = str)
@lightbulb.option("image", "(Optional) image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,default = 30, type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion. Enter \"0\" for random.", required = False, default = 0, type = int, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float,default = 7, max_value=100, min_value=-100)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False,default ="0")
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = False,default ="0")
@lightbulb.command("generate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global titles
    global outputDirectory
    global botBusy
    global outEmbed
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
        thread = threadmanager.New_Thread(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength,ctx.options.width,ctx.options.height,respProxy)
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
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = False)
@lightbulb.command("regenerate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global regentitles
    global outputDirectory
    global botBusy
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
        thread = threadmanager.New_Thread(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength,ctx.options.width,ctx.options.height,respProxy)
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
    "\n> **/changemodel**: switches model between stable diffusion v1.5 or waifu diffusion v1.3"
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
    global prevResultImage
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
@lightbulb.option("model", "which model to load, sd / wd",choices=["sd","wd"],required=True)
@lightbulb.command("changemodel", "switches model between stable diffusion / waifu diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def changemodel(ctx: lightbulb.SlashContext) -> None:
    global pipe
    global curmodel
    global botBusy
    if botBusy:
        await ctx.respond("> Sorry, kiwi is busy!")
        return
    botBusy = True
    if ctx.options.model.startswith("s"):
        await ctx.respond("> **Loading Stable Diffusion v1.5**")
        pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',custom_pipeline="lpw_stable_diffusion",use_auth_token="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",torch_dtype=torch.float16, revision="fp16").to('cuda')
        pipe.enable_attention_slicing()
        await ctx.edit_last_response("> **Model set to Stable Diffusion v1.5**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"
    elif ctx.options.model.startswith("w"):
        await ctx.respond("> **Loading Waifu Diffusion v1.3**")
        pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',custom_pipeline="lpw_stable_diffusion",torch_dtype=torch.float16, revision="fp16").to('cuda')
        pipe.enable_attention_slicing()
        await ctx.edit_last_response("> **Model set to Waifu Diffusion v1.3**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
        ctx.edit_last_response()
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
    if ctx.options.string != None:
        file1 = open("todo.txt","w")
        file1.write(ctx.options.string)
        file1.close()
    file2 = open("todo.txt","r")
    rows = await generate_rows(ctx.bot)
    embed = hikari.Embed(title="Todo:",colour=hikari.Colour(0xabaeff),description=file2.readline().replace(", ",",\n"))
    file2.close()
    response = await ctx.respond(embed,components=rows)
    message = await response.message()
    await handle_responses(ctx.bot, ctx.author, message)

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
#Quit
#-----------------------------  -----
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