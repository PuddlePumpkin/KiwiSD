import lightbulb
import hikari
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


curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
#pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',torch_dtype=torch.float16, revision="fp16").to('cuda')


def filecount():
    return len([entry for entry in os.listdir("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results") if os.path.isfile(os.path.join("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results", entry))])

#----------------------------------
#Globals
#----------------------------------
pipe = None
guideVar = 6.5
infSteps = 30
prevPrompt = ""
awaitingImage = False
awaitingUserId = 0
url = ""
strength = 0.25
imagePrompt = ""
activeMessage = 0
mode = -1
overprocessImage=None
overprocessbool=False
#----------------------------------
#Normal Generate Function
#----------------------------------
def WdGenerate(prompttext):
    global guideVar
    global infSteps
    global mode
    global pipe
    global overprocessImage
    global imagePrompt
    prompt = prompttext
    if (curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"):
        if (mode != 3):
            pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',use_auth_token="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",torch_dtype=torch.float16, revision="fp16").to('cuda')
            mode = 3
    else:
        if (mode != 2):
            pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',torch_dtype=torch.float16, revision="fp16").to('cuda')
            mode = 2
    print("Generating: " + prompttext)
    response = requests.get("https://i.imgur.com/ZMKhIoA.png")
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    with autocast("cuda"):
        def dummy_checker(images, **kwargs): return images, False
        pipe.safety_checker = dummy_checker
        image = pipe(prompt, guidance_scale=guideVar, num_inference_steps=infSteps).images[0]
    countStr = str(filecount()+1)
    while os.path.exists("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png"):
        countStr = int(countStr)+1
    metadata = PngInfo()
    metadata.add_text("Prompt", prompttext)
    imagePrompt=prompttext
    metadata.add_text("Guidance Scale", str(guideVar))
    metadata.add_text("Inference Steps", str(infSteps))
    overprocessImage = image
    image.save("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png", pnginfo=metadata)
    return "C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png"

#----------------------------------
#Image Generate Function
#----------------------------------
def WdGenerateImage(prompttext):
    global guideVar
    global infSteps
    global url
    global strength
    global mode
    global pipe
    global overprocessImage
    global overprocessbool
    prompt = prompttext
    if (curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"):
        if (mode != 1):
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',use_auth_token="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",torch_dtype=torch.float16, revision="fp16").to('cuda')
            mode = 1
    else:
        if (mode != 0):
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained('hakurei/waifu-diffusion',torch_dtype=torch.float16, revision="fp16").to('cuda')
            mode = 0
    print("Generating: " + prompttext)
    if(overprocessbool):
        init_image = overprocessImage
        overprocessbool=False
        print("Got process image from overprocessImage")
    else:
        response = requests.get(url)
        init_image = Image.open(BytesIO(response.content)).convert("RGB")
        init_image = init_image.resize((512, 512))
        print("Got image from url:" + url)
    with autocast("cuda"):
        def dummy_checker(images, **kwargs): return images, False
        pipe.safety_checker = dummy_checker
        image = pipe(prompt, init_image = init_image, strength=(1-strength), guidance_scale=guideVar, num_inference_steps=infSteps).images[0]
    countStr = str(filecount()+1)
    while os.path.exists("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png"):
        countStr = int(countStr)+1
    metadata = PngInfo()
    metadata.add_text("Prompt", prompttext)
    metadata.add_text("Guidance Scale", str(guideVar))
    metadata.add_text("Inference Steps", str(infSteps))
    overprocessImage = image
    image.save("C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png", pnginfo=metadata)
    return "C:/Users/keira/Desktop/GITHUB/Kiwi/venv/Scripts/results/" + str(countStr) + ".png"

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
# Bot Message Event (Process Command)
#----------------------------------
@bot.listen(hikari.GuildMessageCreateEvent)
async def message_received(event):
    global awaitingUserId
    global awaitingImage
    global url
    global imagePrompt
    global pipe
    global mode
    if(awaitingImage):
        if(event.author.id == awaitingUserId):
            try:
                url = event.message.attachments[0].url
                await bot.rest.delete_message(672892614613139471,event.message)
                foundURL=True
            except:
                print("Same user, but no attachment.")
                foundURL=False
            if(foundURL):
                awaitingImage = False
                titles = ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]
                embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = imagePrompt, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
                await bot.rest.edit_message(672892614613139471, activeMessage, embed)
                filepath = WdGenerateImage(imagePrompt)
                f = hikari.File(filepath)
                if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
                    embed.title = "Stable Diffusion v1.5 - Result:"
                else:
                    embed.title = "Waifu Diffusion v1.3 - Result:"
                embed.set_image(f)
                await bot.rest.edit_message(672892614613139471, activeMessage, embed)

#----------------------------------
#Ping Command
#----------------------------------
@bot.command
@lightbulb.command("ping", "checks the bot is alive")
@lightbulb.implements(lightbulb.SlashCommand)
async def ping(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("Pong!")

#----------------------------------
#Process Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ")
@lightbulb.option("strength", "Strength of the input image (Default:0.25)", required = False,type = float, default=0.25)
@lightbulb.command("process", "runs diffusion on an input image")
@lightbulb.implements(lightbulb.SlashCommand)
async def process(ctx: lightbulb.SlashContext) -> None:
    global awaitingUserId
    global awaitingImage
    global strength
    global imagePrompt
    global activeMessage
    global curmodel
    global prevPrompt
    awaitingUserId = ctx.author.id
    embed = hikari.Embed(title="Please send your input image...",colour=hikari.Colour(0xff7e85),).set_footer(text = str(ctx.options.prompt), icon = curmodel)
    responseProxy = await ctx.respond(embed)
    activeMessage = await responseProxy.message()
    awaitingImage = True
    strength = float(ctx.options.strength)
    prevPrompt = str(ctx.options.prompt)
    imagePrompt = str(ctx.options.prompt)

#----------------------------------
#ReProcess Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "(Optional) A detailed description of desired output. Uses last prompt if empty. ",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, default=0.25, max_value=1, min_value=0)
@lightbulb.command("reprocess", "re-runs diffusion on the previous image")
@lightbulb.implements(lightbulb.SlashCommand)
async def reprocess(ctx: lightbulb.SlashContext) -> None:
    global awaitingUserId
    global awaitingImage
    global strength
    global imagePrompt
    global activeMessage
    global curmodel
    titles = ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]
    strength = float(ctx.options.strength)
    if ctx.options.prompt != None:
        imagePrompt = str(ctx.options.prompt)
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8),).set_footer(text = str(imagePrompt), icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
    await ctx.respond(embed)
    filepath = WdGenerateImage(imagePrompt)
    f = hikari.File(filepath)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
    embed.set_image(f)
    await ctx.edit_last_response(embed)

#----------------------------------
#OverProcess Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "(Optional) A detailed description of desired output. Uses last prompt if empty. ",required = False)
@lightbulb.option("strength", "(Optional) Strength of the input image (Default:0.25)", required = False,type = float, default=0.25, max_value=1, min_value=0)
@lightbulb.command("overprocess", "re-runs diffusion on the RESULT of the previous diffusion")
@lightbulb.implements(lightbulb.SlashCommand)
async def overprocess(ctx: lightbulb.SlashContext) -> None:
    global awaitingUserId
    global awaitingImage
    global strength
    global imagePrompt
    global activeMessage
    global curmodel
    global overprocessbool
    titles = ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]
    strength = float(ctx.options.strength)
    if ctx.options.prompt != None:
        imagePrompt = str(ctx.options.prompt)
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8),).set_footer(text = str(imagePrompt), icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
    await ctx.respond(embed)
    overprocessbool=True
    filepath = WdGenerateImage(imagePrompt)
    f = hikari.File(filepath)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
    embed.set_image(f)
    await ctx.edit_last_response(embed)

#----------------------------------
#Help Command
#----------------------------------
@bot.command
@lightbulb.command("help", "get help and command info")
@lightbulb.implements(lightbulb.SlashCommand)
async def help(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("**~~                                                                                    ~~ Generation ~~                                                                                        ~~**\n> **/generate**: Generates a image from a detailed description, or booru tags separated by commas\n> **/regenerate**: Re-generates last entered prompt\n> **/process**: Diffuses from an input image\n> **/reprocess**: Reproccesses last input image\n> **/overprocess**: Diffuses from last diffusion result\n**~~                                                                                       ~~ Settings ~~                                                                                           ~~**\n> **/setsteps**: sets step count, higher for better quality but slower processing.(default:50)\n> **/setguidance**: sets guidance scale, how much the image is influenced by the prompted tags.(default:6.5)\n> **/changemodel**: switches model between stable diffusion v1.5 or waifu diffusion v1.3\n**~~                                                                                         ~~ Other ~~                                                                                              ~~**\n> **/lastprompt**: Prints out last used prompt\n> **/deletelast**: Deletes the last bot message in this channel, for deleting nsfw without admin perms.\n> **/ping**: Checks connection\n**~~                                                                                           ~~ Tips ~~                                                                                               ~~**\n> More tags usually results in better images, especially composition tags.\n> __[Composition Tags](https://danbooru.donmai.us/wiki_pages/tag_group:image_composition)__\n> __[Tag Groups](https://danbooru.donmai.us/wiki_pages/tag_groups)__\n> __[Waifu Diffusion 1.3 Release Notes](https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1)__")

#----------------------------------
#Generate Command
#----------------------------------
@bot.command
@lightbulb.option("prompt", "A detailed description of desired output, or booru tags, separated by commas. ")
@lightbulb.command("generate", "Generate a diffusion image from description or tags, separated by commas")
@lightbulb.implements(lightbulb.SlashCommand)
async def generate(ctx: lightbulb.SlashContext) -> None:
    global prevPrompt
    titles = ["I'll try to make that for you!...", "Maybe I could make that...", "I'll try my best!...", "This might be tricky to make..."]
    embed = hikari.Embed(title=random.choice(titles),colour=hikari.Colour(0x56aaf8)).set_footer(text = ctx.options.prompt, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
    await ctx.respond(embed)
    prevPrompt = ctx.options.prompt
    filepath = WdGenerate(ctx.options.prompt)
    f = hikari.File(filepath)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
    embed.set_image(f)
    await ctx.edit_last_response(embed)

#----------------------------------
#Regenerate Command
#----------------------------------
@bot.command
@lightbulb.command("regenerate", "regenerates last prompt")
@lightbulb.implements(lightbulb.SlashCommand)
async def regenerate(ctx: lightbulb.SlashContext) -> None:
    global prevPrompt
    titles = ["I'll try again!... <:scootcry:1033114138366443600>", "Sorry if I didnt do good enough... <:scootcry:1033114138366443600>", "I'll try my best to do better... <:scootcry:1033114138366443600>"]
    embed = hikari.Embed(
            title=random.choice(titles),
            colour=hikari.Colour(0x56aaf8),
            #timestamp=datetime.datetime.now().astimezone()
            ).set_footer(text = prevPrompt, icon = curmodel).set_image("https://i.imgur.com/ZCalIbz.gif")
    await ctx.respond(embed)
    filepath = WdGenerate(prevPrompt)
    f = hikari.File(filepath)
    if curmodel == "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png":
        embed.title = "Stable Diffusion v1.5 - Result:"
    else:
        embed.title = "Waifu Diffusion v1.3 - Result:"
    embed.set_image(f)
    await ctx.edit_last_response(embed)

#----------------------------------
#Set Guidance Command
#----------------------------------
@bot.command
@lightbulb.option("value", "default: 6.5 (how much the result is influenced)",type=float, min_value=-100, max_value=100, default=6.5)
@lightbulb.command("setguidance", "Sets the diffusion guidance scale (tag influence)(recommended between 6 and 9)")
@lightbulb.implements(lightbulb.SlashCommand)
async def setguidance(ctx: lightbulb.SlashContext) -> None:
    global guideVar
    try:
        await ctx.respond("> Set the guidance scale from **" + str(guideVar) + "** to **" + str(float(ctx.options.value)) + "** (Default: **6.5**)")
        guideVar = float(ctx.options.value)
    except ValueError:
        await ctx.respond("Failed to set guidance scale. Current Value: " + str(guideVar))

#----------------------------------
#Set Steps Command
#----------------------------------
@bot.command
@lightbulb.option("value", "default: 30 (how many inference steps to use)",type=int, min_value = 1, max_value = 100, default=30)
@lightbulb.command("setsteps", "inf step count, effects image quality. (recommended between 25 and 75)")
@lightbulb.implements(lightbulb.SlashCommand)
async def setsteps(ctx: lightbulb.SlashContext) -> None:
    global infSteps
    try:
        await ctx.respond("> Set the step count from **" + str(infSteps) + "** to **" + str(int(ctx.options.value)) + "** (Default: **30**)")
        infSteps = int(ctx.options.value)
    except ValueError:
        await ctx.respond("Failed to set step count. Current Value: " + str(infSteps))

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
            await ctx.respond(f"> **I deleted the last message I sent...**\n> I'm sorry if it was too lewd >///<!")
            #else:
            #await ctx.respond(f"Could not find any message in the array!")
    else:
        await ctx.respond("Sorry >~< I couldnt find any messages I sent recently!")

#----------------------------------
#Last Prompt
#----------------------------------
@bot.command()
@lightbulb.command("lastprompt", "print out the last prompt (used for /regenerate).")
@lightbulb.implements(lightbulb.SlashCommand)
async def lastprompt(ctx: lightbulb.SlashContext) -> None:
    global prevPrompt
    await ctx.respond("> **Last prompt used:** \n > " + prevPrompt)

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
    global mode
    if ctx.options.model.startswith("s"):
        await ctx.respond("> **Loading stable diffusion...**")
        pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',use_auth_token="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",torch_dtype=torch.float16, revision="fp16").to('cuda')
        mode = 3
        await ctx.edit_last_response("> **Loaded Stable Diffusion v1.5!**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266027798528/SD-01.png"
    elif ctx.options.model.startswith("w"):
        await ctx.respond("> **Loading waifu diffusion...**")
        pipe = StableDiffusionPipeline.from_pretrained('hakurei/waifu-diffusion',torch_dtype=torch.float16, revision="fp16").to('cuda')
        mode = 2
        await ctx.edit_last_response("> **Loaded Waifu Diffusion v1.3!**")
        curmodel = "https://cdn.discordapp.com/attachments/672892614613139471/1034513266719866950/WD-01.png"
    else:
        await ctx.respond("> **I don't understand** <:scootcry:1033114138366443600>")

#----------------------------------
#Quit
#----------------------------------
@bot.command()
@lightbulb.command("quit", "Puts kiwi to sleep")
@lightbulb.implements(lightbulb.SlashCommand)
async def quit(ctx: lightbulb.SlashContext) -> None:
    await ctx.respond("> **I'm gonna take a little nappy now... see you later... 💤**")
    await bot.close()
    await asyncio.sleep(1)
    await quit(1)

bot.run()
sys.exit()