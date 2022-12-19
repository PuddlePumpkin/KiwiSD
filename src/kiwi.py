import json
import os
import shutil
import sys
import traceback
import datetime
from pathlib import Path

import hikari
import lightbulb
import warnings

os.chdir(str(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))))
config = {}
# ----------------------------------
# Configs
# ----------------------------------
def save_user_config(userid: str, saveconfig):
    '''Saves a user setting to the json file'''
    with open('usersettings.json', 'r') as openfile:
        userconfigs = json.load(openfile)
        userconfigs[str(userid)] = saveconfig
        openfile.close()
    with open("usersettings.json", "w") as outfile:
        json.dump(userconfigs, outfile, indent=4)
        outfile.close()

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

async def generate_role_selection_row(bot:lightbulb.BotApp):
    rows = []
    row = bot.rest.build_action_row()
    menu = row.add_select_menu("RoleMenu")
    for item in config["RolesToToggle"].replace(", ", ",").replace(" ,", ",").replace(" , ", ",").split(","):
        menu.add_option(item,item).add_to_menu()
    menu.set_placeholder("Role")
    menu.add_to_container()
    rows.append(row)
    return rows

async def handle_role_selection(bot: lightbulb.BotApp) -> None:
    """Watches for events, and handles responding to them."""
    with bot.stream(hikari.InteractionCreateEvent,timeout=None).filter(lambda e: (isinstance(e.interaction, hikari.ComponentInteraction))) as stream:
        async for event in stream:
            load_config()
            member = await bot.rest.fetch_member(guildId,event.interaction.user)
            roles = member.get_roles()
            matchedRole = False
            try:
                for obj in config["IncompatibleRoles"]:
                    for key in obj.keys():
                        val = obj[key]
                        if event.interaction.values[0] == key:
                            for role in roles:
                                if role.name == val:
                                    await event.interaction.create_initial_response(hikari.ResponseType.MESSAGE_CREATE, "The selected role is incompatible with one of your roles...", flags=hikari.MessageFlag.EPHEMERAL)
                                    await rolechangemessage()
                                    return
                        elif event.interaction.values[0] == val:
                            for role in roles:
                                if role.name == key:
                                    await event.interaction.create_initial_response(hikari.ResponseType.MESSAGE_CREATE, "The selected role is incompatible with one of your roles...", flags=hikari.MessageFlag.EPHEMERAL)
                                    await rolechangemessage()
                                    return
            except Exception:
                traceback.print_exc()
            for role in roles:
                if role.name == event.interaction.values[0]:
                    matchedRole = True
            roleList = await bot.rest.fetch_roles(guildId)
            for role in roleList:
                if role.name == event.interaction.values[0]:
                    matchedRoleId = role.id
                    break
            if matchedRole:
                await member.remove_role(matchedRoleId)
            else:
                await member.add_role(matchedRoleId)
            await event.interaction.create_initial_response(hikari.ResponseType.MESSAGE_CREATE, "The selected role has been toggled.", flags=hikari.MessageFlag.EPHEMERAL)
            await rolechangemessage()
            

# ----------------------------------
# Instantiate a Bot instance
# ----------------------------------
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
    if tokendict["guildID"] != None and tokendict["guildID"] != "":
        guildId = int(tokendict["guildID"])
    else: 
        warnings.warn("Commands will not update quickly without a guild ID in kiwitoken.json",UserWarning)
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
    await rolechangemessage()
    

async def rolechangemessage():
    load_config()
    try:
        if config["RoleChangeChannel"] != "" and config["RolesToToggle"] != "":
            embed = hikari.Embed(title=("To toggle a role, select it from the list:         "))
            embed.color = hikari.Colour(0xfd87ff)
            rows = await generate_role_selection_row(bot)
            messages = await bot.rest.fetch_messages(int(config["RoleChangeChannel"])).take_until(lambda m: datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30) > m.created_at).limit(1)
            try:
                await bot.rest.edit_message(int(config["RoleChangeChannel"]),messages[0].id,embed,components=rows)
            except:
                await bot.rest.create_message(int(config["RoleChangeChannel"]),embed,components=rows)
            await handle_role_selection(bot)
    except Exception:
        traceback.print_exc()
        

async def respond_with_autodelete(text: str, ctx: lightbulb.SlashContext, color=0xff0015):
    '''Generate an embed and respond to the context with the input text'''
    await ctx.respond(text, flags=hikari.MessageFlag.EPHEMERAL)

def set_bool_user_setting(userid: str, setting: str, value: bool):
    '''Sets a user setting to specified bool'''
    userconfig = load_user_config(userid)
    userconfig[setting] = value
    save_user_config(userid, userconfig)

def get_bool_user_setting(userid: str, setting: str) -> bool:
    '''Gets a bool user setting'''
    userconfig = load_user_config(userid)
    return userconfig[setting]

# ----------------------------------
# Start Bot
# ----------------------------------
bot.run()
sys.exit()
