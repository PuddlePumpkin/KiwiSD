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
from torch import autocast, negative
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
prevSeed = None
prevResultImage = None
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
#Footer
#----------------------------------
def get_embed(Prompt,NegativePrompt, GuideScale, InfSteps, Seed, File, ImageStrength=None):
    global curmodel
    if ImageStrength != None:
        footer = ("{:28s} {:28s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:28s} {:28s}".format("Seed: "+str(Seed), "Image Strength: "+str(ImageStrength)))
    else:
        footer = ("{:28s} {:28s}".format("Guidance Scale: "+str(GuideScale), "Inference Steps: "+str(InfSteps))+"\n"+"{:28s}".format("Seed: "+str(Seed)))
    f = hikari.File(File)
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = footer, icon = curmodel).set_image(f)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
    embed.add_field("Prompt:",Prompt)
    if ((NegativePrompt != None) and (NegativePrompt!= "None") and (NegativePrompt!= "")):
        embed.add_field("Negative Prompt:",prevNegPrompt)
    return embed
    
#----------------------------------
#Generate Function
#----------------------------------
def WdGenerateImage(Prompt=None,NegativePrompt=None,InfSteps=None,Seed=None,GuideScale=None,ImgUrl=None,Strength=None):
    global prevPrompt
    global prevNegPrompt
    global prevInfSteps
    global prevGuideScale
    global prevSeed
    global prevUrl
    global prevStrength
    global prevResultImage

    global pipe
    global overprocessbool

#Handle prompt
    if(Prompt!=None):
        prompt = Prompt
    elif(prevPrompt!=None):
        prompt = prevPrompt
    
#Handle negative prompt
    if(NegativePrompt!=None):
        if(NegativePrompt=="0"):
            prevNegPrompt = None
            negativeprompt = None
        else:
            prevNegPrompt = NegativePrompt
            negativeprompt = NegativePrompt
    #Its okay to use prev neg prompt even if empty
    else:
        negativeprompt = prevNegPrompt

#Handle infsteps
    if(InfSteps!=None):
        if(InfSteps=="0"):
            prevInfSteps = 30
            infsteps = 30
        else:
            infsteps = InfSteps
    elif(prevInfSteps!=None):
        infsteps = prevInfSteps
    elif(prevInfSteps==None):
        prevInfSteps = 30
        infsteps = prevInfSteps

#Handle guidescale
    if(GuideScale!=None):
        if(GuideScale=="0"):
            prevGuideScale = 7.0
            guidescale = 7.0
        else:
            guidescale = GuideScale
    elif(prevGuideScale!=None):
        guidescale = prevGuideScale
    elif(prevGuideScale==None):
        prevGuideScale = 7
        guidescale = prevGuideScale

#Handle Stength
    if(Strength!=None):
        if(Strength=="0"):
            prevStrength = 0.25
            strength = 0.25
        else:
            strength = Strength
    elif(prevStrength!=None):
        strength = prevStrength
    elif(prevStrength==None):
        prevStrength = 0.25
        strength = prevStrength

#Handle ImgUrl
    if(ImgUrl!=None):
        if(ImgUrl=="0"):
            prevUrl = None
            prevStrength = None
            strength = None
            imgurl = None
        else:
            imgurl = ImgUrl
    elif(prevUrl!=None):
        imgurl = prevUrl
    elif(prevUrl==None):
        prevUrl = None
        imgurl = prevUrl

#Handle seed
    if(Seed!=None):
        if(Seed==0):
            prevSeed = random.randint(1,100000000)
            seed = prevSeed
            generator = torch.Generator("cuda").manual_seed(prevSeed)
        else:
            prevSeed = Seed
            seed = Seed
            generator = torch.Generator("cuda").manual_seed(Seed)
    elif(prevSeed!=None):
        seed = prevSeed
        generator = torch.Generator("cuda").manual_seed(prevSeed)
    elif(prevSeed==None):
        prevSeed = random.randint(1,100000000)
        seed = prevSeed
        generator = torch.Generator("cuda").manual_seed(prevSeed)


    print("Generating: " + prompt)
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
    metadata.add_text("Prompt", prompt)
    if ((negativeprompt != None) and (negativeprompt!= "None") and (negativeprompt!= "")):
        metadata.add_text("Negative Prompt", negativeprompt)
    metadata.add_text("Guidance Scale", str(guidescale))
    metadata.add_text("Inference Steps", str(infsteps))
    metadata.add_text("Seed", str(seed))

    with autocast("cuda"):
        def dummy_checker(images, **kwargs): return images, False
        pipe.safety_checker = dummy_checker
        if strength != None:
            image = pipe(prompt, generator = generator, init_image = init_image, negative_prompt=negativeprompt, strength=(1-strength), guidance_scale=guidescale, num_inference_steps=infsteps).images[0]
            metadata.add_text("Img2Img Strength", str(strength))
        else:
            image = pipe(prompt, generator = generator, init_image = init_image, negative_prompt=negativeprompt, guidance_scale=guidescale, num_inference_steps=infsteps).images[0]
    countStr = str(filecount()+1)
    while os.path.exists("C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png"):
        countStr = int(countStr)+1

    prevResultImage = image
    image.save("C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png", pnginfo=metadata)
    return get_embed(prompt,negativeprompt,guidescale,infsteps,seed,"C:/Users/keira/Desktop/GITHUB/Kiwi/results/" + str(countStr) + ".png",strength)

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
#Generate Command
#----------------------------------
@bot.command
@lightbulb.option("image", "image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = True)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False,default = "0")
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, default=0.25)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False, default = 7,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, default=30, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0, default = 0)
@lightbulb.command("generate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global titles
    try:
        if(ctx.options.image == None):
            url = "0"
            #strength = "0"
        else:
            url = ctx.options.image.url
            #strength = ctx.options.strength
        #--Embed                  
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = "", icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        await ctx.respond(embed)
        #-------
        embed = WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        return

#----------------------------------
#Generate From Image Command
#----------------------------------
@bot.command
@lightbulb.option("image", "image to run diffusion on", required = True,type = hikari.Attachment)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = True)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False,default = "0")
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, default=0.25)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False, default = 7,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, default=30, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0, default = 0)
@lightbulb.command("genfromimage", "shortcut for /generate, requires an image to make quickly entering images easier")
@lightbulb.implements(lightbulb.SlashCommand)
async def genfromimage(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global titles
    try:
        if(ctx.options.image == None):
            url = "0"
            #strength = "0"
        else:
            url = ctx.options.image.url
            #strength = ctx.options.strength
        #--Embed                  
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = "", icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        await ctx.respond(embed)
        #-------
        embed = WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
        return

#----------------------------------
#ReGenerate Command
#----------------------------------
@bot.command
@lightbulb.option("image", "image to run diffusion on", required = False,type = hikari.Attachment)
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ",required = False)
@lightbulb.option("negativeprompt", "(Optional)Prompt for diffusion to avoid.",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float)
@lightbulb.option("guidescale", "(Optional) Guidance scale for diffusion (Default:7)", required = False,type = float, max_value=100, min_value=-100)
@lightbulb.option("steps", "(Optional) Number of inference steps to use for diffusion (Default:30)", required = False,type = int, max_value=100, min_value=1)
@lightbulb.option("seed", "(Optional) Seed for diffusion", required = False,type = int, min_value=0)
@lightbulb.command("regenerate", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    global curmodel
    global titles
    try:
        if(ctx.options.image == None):
            url = None
        else:
            url = ctx.options.image.url

        #--Embed                  
        embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = "", icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
        await ctx.respond(embed)
        #-------
        embed = WdGenerateImage(ctx.options.prompt,ctx.options.negativeprompt,ctx.options.steps,ctx.options.seed,ctx.options.guidescale,url,ctx.options.strength)
        await ctx.edit_last_response(embed)
    except Exception:
        traceback.print_exc()
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
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
    await bot.rest.create_message(672892614613139471,embedtext2)

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
        embed = hikari.Embed(title="Sorry, something went wrong! <:scootcry:1033114138366443600>",colour=hikari.Colour(0xFF0000))
        await ctx.edit_last_response(embed)
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