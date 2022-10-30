from sqlite3 import Timestamp
import time
import lightbulb
import hikari
from prompt_toolkit import prompt
import torch
import sys
import datetime
import asyncio
import os
import gc
import requests
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from io import BytesIO
import random
import traceback

#----------------------------------
#Globals
#----------------------------------
curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',custom_pipeline="lpw_stable_diffusion",torch_dtype=torch.float16, revision="fp16").to('cuda')
guideVar = 6.5
infSteps = 30
prevPrompt = ""
prevNegPrompt = ""
prevStrength = 0.25
prevUrl = ""
prevSeed = 0
overprocessImage = None
overprocessbool = False
regentitles = ["I'll try again!... <:scootcry:1033114138366443600>", "Sorry if I didnt do good enough... <:scootcry:1033114138366443600>", "I'll try my best to do better... <:scootcry:1033114138366443600>"]
titles =  ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]

#----------------------------------
#Filecount Function
#----------------------------------
def filecount():
    return len([entry for entry in os.listdir("C:/Users/keira/Desktop/GITHUB/Kiwi/results") if os.path.isfile(os.path.join("C:/Users/keira/Desktop/GITHUB/Kiwi/results", entry))])

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
#Generate Function
#----------------------------------
def WdGenerateImage(prompttext, negativeprompttext, processimage = False):
    global guideVar
    global infSteps
    global prevUrl
    global prevStrength
    global prevSeed
    global pipe
    global overprocessImage
    global overprocessbool
    prompt = prompttext
    if prevSeed != None:
        generator = torch.Generator("cuda").manual_seed(prevSeed)
    else:
        generator = None
    print("Generating: " + prompttext)
    if processimage:
        if(overprocessbool):
            print("Loading image from overprocessImage")
            init_image = overprocessImage
            overprocessbool=False
        else:
            print("Loading image from url: " + prevUrl)
            response = requests.get(prevUrl)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            #Crop and resize
            init_image = crop_max_square(init_image)
            init_image = init_image.resize((512, 512),Image.Resampling.LANCZOS)
    else:
        init_image = None
    if ((negativeprompttext != None) and (negativeprompttext!= "None") and (negativeprompttext!= "")):
        metadata.add_text("Negative Prompt", negativeprompttext)
        negativeprompttext = None
    with autocast("cuda"):
        def dummy_checker(images, **kwargs): return images, False
        pipe.safety_checker = dummy_checker
        image = pipe(prompt, generator = generator, init_image = init_image, negative_prompt=negativeprompttext, strength=(1-prevStrength), guidance_scale=guideVar, num_inference_steps=infSteps).images[0]
    countStr = str(filecount()+1)
    while os.path.exists("C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png"):
        countStr = int(countStr)+1
    metadata = PngInfo()
    metadata.add_text("Prompt", prompttext)

    if(processimage):
        metadata.add_text("Img2Img Strength", str(prevStrength))
    metadata.add_text("Guidance Scale", str(guideVar))
    metadata.add_text("Inference Steps", str(infSteps))
    overprocessImage = image
    image.save("C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png", pnginfo=metadata)
    return "C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png"

#----------------------------------    
# Instantiate a Bot instance
#----------------------------------
bot = lightbulb.BotApp(
    token="***REMOVED***",
    prefix="-",
    default_enabled_guilds=672718048343490581,
    intents=hikari.Intents.ALL,
    help_class=None
    )

#----------------------------------    
# Bot Ready Event
#----------------------------------
@bot.listen(hikari.ShardReadyEvent)
async def ready_listener(_):
    await bot.rest.create_message(672892614613139471, "> **I'm awake running waifu diffusion v1.3! Type /help for help!**")


#----------------------------------
#Ping Command
#----------------------------------
@bot.command
@lightbulb.command("ping", "checks the bot is alive")
@lightbulb.implements(lightbulb.SlashCommand)
async def ping(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("Pong!")

#----------------------------------
#Metadata Command
#----------------------------------
@bot.command
@lightbulb.option("image", "input image", required = True,type = hikari.Attachment)
@lightbulb.command("metadata", "check metadata of an image")
@lightbulb.implements(lightbulb.SlashCommand)
async def metadata(ctx: lightbulb.SlashContext) -> None:
    datas = await hikari.Attachment.read(ctx.options.image)
    mdataimage = Image.open(BytesIO(datas)).convert("RGB")
    mdataimage = mdataimage.resize((512, 512))
    embed = hikari.Embed(title=(ctx.options.image.url.rsplit('/', 1)[-1]),colour=hikari.Colour(0x56aaf8)).set_thumbnail(ctx.options.image.url)
    if(str(mdataimage.info.get("Prompt")) != "None"):
        embed.add_field("Prompt:",str(mdataimage.info.get("Prompt")))
    if(str(mdataimage.info.get("Negative Prompt")) != "None"):
        embed.add_field("Negative Prompt:",str(mdataimage.info.get("Negative Prompt")))
    if(str(mdataimage.info.get("Guidance Scale")) != "None"):
        embed.add_field("Guidance Scale:",str(mdataimage.info.get("Guidance Scale")))
    if(str(mdataimage.info.get("Inference Steps")) != "None"):
        embed.add_field("Inference Steps:",str(mdataimage.info.get("Inference Steps")))
    if(str(mdataimage.info.get("Img2Img Strength")) != "None"):
        embed.add_field("Img2Img Strength:",str(mdataimage.info.get("Img2Img Strength")))
    await ctx.respond(embed)

#----------------------------------
#Process Command
#----------------------------------
@bot.command
@lightbulb.option("image", "image to run diffusion on", required = True,type = hikari.Attachment)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = True)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, default=0.25)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False, default = 7,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, default=30, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("process", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def process(ctx: lightbulb.SlashContext) -> None:
    global prevStrength
    global curmodel
    global prevPrompt
    global prevNegPrompt
    global prevUrl
    global guideVar
    global titles
    global infSteps
    global prevSeed
    try:
        #--Inputs
        prevSeed = ctx.options.seed
        prevStrength = float(ctx.options.strength)
        prevPrompt = str(ctx.options.prompt)
        prevNegPrompt = str(ctx.options.negativeprompt)
        guideVar = float(ctx.options.guidescale)
        infSteps = int(ctx.options.steps)

        #--------
        
        prevUrl = ctx.options.image.url

        #--Embed
        footer = "Guidance Scale: " + str(guideVar) + "                 Inference Steps: "+str(infSteps) +"\nImage Strength: "+str(prevStrength)                   
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        embed.add_field("Prompt:",prevPrompt)
        if ctx.options.negativeprompt != None:
            embed.add_field("Negative Prompt:",prevNegPrompt)
        await ctx.respond(embed)
        #-------

        filepath = WdGenerateImage(prevPrompt,prevNegPrompt,True)
        f = hikari.File(filepath)
        if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
            embed.title = "Stable Diffusion v1.5 - Result:"
        else:
            embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.set_image(f)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        try:
            await ctx.delete_last_response()
        except Exception:
            traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return

#----------------------------------
#Reprocess Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "(Optional) A detailed description of desired output. Uses last prompt if empty. ",required = False)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, max_value=1, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("reprocess", "re-runs diffusion on the previous input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def reprocess(ctx: lightbulb.SlashContext) -> None:
    global prevStrength
    global curmodel
    global prevNegPrompt
    global prevPrompt
    global guideVar
    global regentitles
    global infSteps
    global prevSeed
    try:
        #--Inputs
        if ctx.options.seed != None:
            if ctx.options.seed == 0:
                prevSeed = None
            else:
                prevSeed = ctx.options.seed
        if ctx.options.strength != None:
            prevStrength = float(ctx.options.strength)
        if ctx.options.prompt != None:
            prevPrompt = str(ctx.options.prompt)
        if ctx.options.negativeprompt != None:
            prevNegPrompt = str(ctx.options.negativeprompt)
        if ctx.options.guidescale != None:
            guideVar = float(ctx.options.guidescale)
        if ctx.options.steps != None:
            infSteps = int(ctx.options.steps)
        #--------

        #--Embed
        footer = "Guidance Scale: " + str(guideVar) + "                 Inference Steps: "+str(infSteps) +"\nImage Strength: "+str(prevStrength)
        embed = hikari.Embed(title=random.choice(regentitles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        embed.add_field("Prompt:",prevPrompt)
        if ((prevNegPrompt != None) and (prevNegPrompt!= "None") and (prevNegPrompt!= "")):
            embed.add_field("Negative Prompt:",prevNegPrompt)
        else:
            prevNegPrompt = ""
        await ctx.respond(embed)
        #-------

        filepath = WdGenerateImage(prevPrompt,prevNegPrompt,True)
        f = hikari.File(filepath)
        if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
            embed.title = "Stable Diffusion v1.5 - Result:"
        else:
            embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.set_image(f)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        try:
            await ctx.delete_last_response()
        except Exception:
            traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return

#----------------------------------
#OverProcess Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "(Optional) A detailed description of desired output. Uses last prompt if empty. ",required = False)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, max_value=1, min_value=0)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("overprocess", "re-runs diffusion on the RESULT of the previous diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def overprocess(ctx: lightbulb.SlashContext) -> None:
    global prevStrength
    global curmodel
    global overprocessbool
    global prevPrompt
    global prevNegPrompt
    global guideVar
    global regentitles
    global infSteps
    global prevSeed
    try:
        #--Inputs
        if ctx.options.seed != None:
            if ctx.options.seed == 0:
                prevSeed = None
            else:
                prevSeed = ctx.options.seed
        if ctx.options.strength != None:
            prevStrength = float(ctx.options.strength)
        if ctx.options.prompt != None:
            prevPrompt = str(ctx.options.prompt)
        if ctx.options.negativeprompt != None:
            prevNegPrompt = str(ctx.options.negativeprompt)
        if ctx.options.guidescale != None:
            guideVar = float(ctx.options.guidescale)
        if ctx.options.steps != None:
            infSteps = int(ctx.options.steps)
            
        #--------
        
        #--Embed
        footer = "Guidance Scale: " + str(guideVar) + "                 Inference Steps: "+str(infSteps) +"\nImage Strength: "+str(prevStrength)
        embed = hikari.Embed(title=random.choice(regentitles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        embed.add_field("Prompt:",prevPrompt)
        if ((prevNegPrompt != None) and (prevNegPrompt!= "None") and (prevNegPrompt!= "")):
            embed.add_field("Negative Prompt:",prevNegPrompt)
        else:
            prevNegPrompt = ""
        await ctx.respond(embed)
        #-------

        overprocessbool=True
        filepath = WdGenerateImage(prevPrompt,prevNegPrompt,True)
        f = hikari.File(filepath)
        if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
            embed.title = "Stable Diffusion v1.5 - Result:"
        else:
            embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.set_image(f)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        try:
            await ctx.delete_last_response()
        except Exception:
            traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return

#----------------------------------
#Help Command
#----------------------------------
@bot.command
@lightbulb.command("help", "get help and command info")
@lightbulb.implements(lightbulb.SlashCommand)
async def help(ctx: lightbulb.SlashContext) -> None:
    embedtext1 = (
    "**~~                   ~~ Generation ~~                   ~~**"
    "\n> **/generate**: Generates a image from a detailed description, or booru tags separated by commas"
    "\n> **/regenerate**: Re-generates last entered prompt"
    "\n> **/process**: Diffuses from an input image"
    "\n> **/reprocess**: Reproccesses last input image"
    "\n> **/overprocess**: Diffuses from last diffusion result"
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
    await bot.rest.create_message(672892614613139471,embedtext2)

#----------------------------------
#Generate Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ")
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, default=30, max_value=100, min_value=1)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, default=7, max_value=100, min_value=-100)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("generate", "Generate a diffusion image from description or tags, separated by commas")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global prevPrompt
    global prevNegPrompt
    global infSteps
    global guideVar
    global titles
    global prevSeed
    try:
        #--Inputs
        prevSeed = ctx.options.seed
        prevPrompt = str(ctx.options.prompt)
        prevNegPrompt = str(ctx.options.negativeprompt)
        infSteps = int(ctx.options.steps)
        guideVar = float(ctx.options.guidescale)
        #-------

        #--Embed
        footer = "Guidance Scale: " + str(guideVar) + "                 Inference Steps: "+str(infSteps)                   
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        embed.add_field("Prompt:",prevPrompt)
        if ctx.options.negativeprompt != None:
            embed.add_field("Negative Prompt:",prevNegPrompt)
        await ctx.respond(embed)
        #-------

        filepath = WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt)
        f = hikari.File(filepath)
        if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
            embed.title = "Stable Diffusion v1.5 - Result:"
        else:
            embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.set_image(f)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        try:
            await ctx.delete_last_response()
        except Exception:
            traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return


#----------------------------------
#Admin Generate Gif
#----------------------------------
@bot.command
@lightbulb.add_checks(lightbulb.owner_only)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ")
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:20)", required = False,type = int, default=20, max_value=100, min_value=1)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, default=7, max_value=100, min_value=-100)
@lightbulb.option("animatedkey", "which key (guidescale, steps, strength)", required = True,type = str)
@lightbulb.option("animatedstep", "step value", required = True,type = float)
@lightbulb.option("animatedstart", "start value", required = True,type = float)
@lightbulb.option("animatedend", "end value", required = True,type = float)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,default=0.25,type = float, max_value=1, min_value=0)
@lightbulb.option("image", "image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("admingenerategif", "Generate a series of results")
@lightbulb.implements(lightbulb.SlashCommand)
async def admingenerategif(ctx: lightbulb.SlashContext) -> None:
    global guideVar
    global infSteps
    global prevUrl
    global prevStrength
    global prevSeed
    prevSeed = ctx.options.seed
    try:
        await ctx.respond("> Running gif generation...")
        guideVar = ctx.options.guidescale
        infSteps = ctx.options.steps
        if ctx.options.image != None:
            prevUrl = ctx.options.image.url
        curstep = ctx.options.animatedstart
        prevStrength = ctx.options.strength
        prevNegPrompt = str(ctx.options.negativeprompt)
        startTime = time.time()
        if ctx.options.animatedend > ctx.options.animatedstart:
            if ctx.options.animatedstep > 0:
                while curstep <= ctx.options.animatedend:
                    curstep = curstep + ctx.options.animatedstep
                    if ctx.options.animatedkey == "guidescale":
                        guideVar = float(curstep)
                    elif ctx.options.animatedkey == "steps":
                        infSteps = int(curstep)
                    elif ctx.options.animatedkey == "strength":
                        if ((prevStrength + float(ctx.options.animatedstep))<=1):
                            prevStrength = curstep 
                        else:
                            prevStrength = 1
                    if ctx.options.image != None:
                        WdGenerateImage(ctx.options.prompt,str(ctx.options.negativeprompt,True))
                    else:
                        WdGenerateImage(ctx.options.prompt,str(ctx.options.negativeprompt))
                    if (time.time()-startTime >= 10):
                        await ctx.edit_last_response("> Animation progress: **" + str(int(curstep/(ctx.options.animatedend-ctx.options.animatedstart)*100))+"%**")
                        startTime = time.time()
            else:
                raise Exception("step not matching")
        else:
            if ctx.options.animatedstep < 0:
                while curstep >= ctx.options.animatedend:
                    curstep = curstep + ctx.options.animatedstep
                    if ctx.options.animatedkey == "guidescale":
                        guideVar = float(curstep)
                    elif ctx.options.animatedkey == "steps":
                        infSteps = int(curstep)
                    elif ctx.options.animatedkey == "strength":
                        if ((prevStrength + float(ctx.options.animatedstep))>=0):
                            prevStrength = curstep 
                        else:
                            prevStrength = 0
                    WdGenerateImage(ctx.options.prompt,str(ctx.options.negativeprompt),True)
                    if (time.time()-startTime >= 10):
                        await ctx.edit_last_response("> Animation progress: **" + str(int(1-(curstep/(ctx.options.animatedend-ctx.options.animatedstart)*100)))+"%**")
                        startTime = time.time()
            else:
                raise Exception("step not matching: " + str(ctx.options.animatedstep))
        await ctx.edit_last_response("> Animation Complete.")
    except Exception:
        traceback.print_exc()
        try:
            await ctx.delete_last_response()
        except Exception:
            traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return

#----------------------------------
#Regenerate Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = False)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, max_value=100, min_value=1)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, max_value=100, min_value=-100)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("regenerate", "regenerates last prompt")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    global prevPrompt
    global prevNegPrompt
    global infSteps
    global guideVar
    global regentitles
    global prevSeed
    #try for if no prompt to regen
    try:
        #--Inputs
        if ctx.options.seed != None:
            if ctx.options.seed == 0:
                prevSeed = None
            else:
                prevSeed = ctx.options.seed
        if ctx.options.prompt != None:
            prevPrompt = str(ctx.options.prompt)
        if ctx.options.negativeprompt != None:
            prevNegPrompt = str(ctx.options.negativeprompt)
        if ctx.options.steps != None:
            infSteps = int(ctx.options.steps)
        if ctx.options.guidescale != None:
            guideVar = float(ctx.options.guidescale)
        #--------
    

        #--Embed
        footer = "Guidance Scale: " + str(guideVar) + "                 Inference Steps: "+str(infSteps)                   
        embed = hikari.Embed(title=random.choice(regentitles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        embed.add_field("Prompt:",prevPrompt)
        if ((prevNegPrompt != None) and (prevNegPrompt!= "None") and (prevNegPrompt!= "")):
            embed.add_field("Negative Prompt:",prevNegPrompt)
        else:
            prevNegPrompt = ""
        await ctx.respond(embed)
        #-------

        filepath = WdGenerateImage(prevPrompt,prevNegPrompt)
        f = hikari.File(filepath)
        if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
            embed.title = "Stable Diffusion v1.5 - Result:"
        else:
            embed.title = "Waifu Diffusion v1.3 - Result:"
        embed.set_image(f)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        await ctx.respond("> Sorry, something went wrong! <:scootcry:1033114138366443600>")
        return

#----------------------------------
#Delete Last Command
#----------------------------------
@bot.command()
@lightbulb.command("deletelast", "delete previous message in channel.")
@lightbulb.implements(lightbulb.SlashCommand)
async def deletelast(ctx: lightbulb.SlashContext) -> None:
    """Purge a certain amount of messages from a channel."""
    if not ctx.guild_id:
        await ctx.respond("This command can only be used in a server.")
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

#----------------------------------
#Change Model
#----------------------------------
@bot.command()
@lightbulb.option("model", "which model to load, sd / wd")
@lightbulb.command("changemodel", "switches model between stable diffusion / waifu diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def changemodel(ctx: lightbulb.SlashContext) -> None:
    global pipe
    global curmodel
    if ctx.options.model.startswith("s"):
        await ctx.respond("> **Loading Stable Diffusion v1.5**")
        pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',custom_pipeline="lpw_stable_diffusion",use_auth_token="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",torch_dtype=torch.float16, revision="fp16").to('cuda')
        await ctx.edit_last_response("> **Model set to Stable Diffusion v1.5**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"
    elif ctx.options.model.startswith("w"):
        await ctx.respond("> **Loading Stable Diffusion v1.5**")
        pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',custom_pipeline="lpw_stable_diffusion",torch_dtype=torch.float16, revision="fp16").to('cuda')
        await ctx.edit_last_response("> **Model set to Waifu Diffusion v1.3**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
    else:
        await ctx.respond("> **I don't understand** <:scootcry:1033114138366443600>")

#----------------------------------
#Quit
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