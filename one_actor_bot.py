import asyncio
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import logging
from aiogram import Bot, Dispatcher
from aiogram import F as aif
import json

from aiogram.dispatcher.router import Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton
from aiogram.types import Message, ReplyKeyboardRemove, FSInputFile, CallbackQuery, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.formatting import Text
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.dispatcher.router import Router
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.fsm.storage.memory import MemoryStorage
from time import sleep
import uuid

API_TOKEN = 'YOUT_TOKEN'

CREATE_CHARACTER = "Create character"
SHOW_CHARACTERS = "Show characters"
CREATE_ILLUSTRATION = "Create illustration"
HELP = "Help"
START = "Start"
REDO = "Regenerate illustration"

from OneActor.generate_data_target import generate_target
from OneActor.generate_data_base import generate_base
from OneActor.tune_mask import tune
from OneActor.inference_mask import inference

# def generate_target(*args, **kwargs):
#     sleep(0.01)
# def generate_base(*args, **kwargs):
#     sleep(0.01)
# def tune(*args, **kwargs):
#     sleep(0.01)
# def inference(*args, **kwargs):
#     sleep(0.01)


bot = Bot(token=API_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

HELP_MESSAGE = """
I will create an illustration for your story.
First create the main character and then describe the illustration.
    
Commands which can help: 
/start – start
/help – list of commands
/new_character – create new character
/new_illustration – create new illustration
/show_characters – show all previously created characters
"""

SMALL_HELP_MESSAGE = """
To create an illustration use /new_illustration
To create a new character use /new_character
Or displayall created characters with /show_characters
"""

user_messages = {}
users_state = {}
characters = []

UNDEFINED = 0
NEW_CHARACTER = 1
NEW_ILLUSTRATION = 2

class CharacterStore:
    def __init__(self, base_concept):
        self.base_concept = base_concept.lower()
        self.subject = None
        self.target_image_path = None
        self.base_all_image_path = None
        self.exp_path = None
        self.model_path = None
        
    def make_json(self):
        res = {}
        for attribute, value in self.__dict__.items():
            res[attribute] = value
        return res
    
class UserState:
    def __init__(self):
        self.state = None
        self.character = None
        self.user_messages = []



def get_n_characters():
    created_characters = [crt for crt in characters if crt.model_path is not None]
    return len(created_characters)

def get_name_from_message(message: Message) -> str:
    """
    Get User Name from any object (user or chat)
    """
    res = ""
    if message.chat.username is not None:
        res += message.chat.username
    if message.chat.full_name is not None:
        res = ("" if res == "" else res + " ") + message.chat.full_name
    if res == "":
        res = f"user_id: {message.chat.id}"
    return res

def save_characters():
    res = {"characters": [crt.make_json() for crt in characters]}
    with open("/workspace/experiments/characters.json", "w") as f:
        json.dump(res, f, ensure_ascii=False)

def load_caracters():
    with open("/workspace/experiments/characters.json", "r") as f:
        res = json.load(f)
    for r in res['characters']:
        characters.append(CharacterStore(r["base_concept"]))
        for attribute, value in characters[-1].__dict__.items():
            setattr(characters[-1], attribute, r[attribute])        
    print(characters)
    


@router.message(aif.text, Command(START))
async def send_welcome(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command {START}")
    if user_id not in users_state:
        users_state[user_id] = UserState()
    logger.info(f"User {message.from_user.id} started the bot")

    n_characters = get_n_characters()
    keyboard_buttons = []
    _message = "Welcome! I'm *DrawStory* - an application to help you create illustrations for your story.\n\n"
    keyboard_buttons.append(KeyboardButton(text=CREATE_CHARACTER))
    _message += f"*{CREATE_CHARACTER}* to create a new character\n"

    if users_state[user_id].character is not None:
        keyboard_buttons.append(KeyboardButton(text=CREATE_ILLUSTRATION))
        _message +=  f"*{CREATE_ILLUSTRATION}* to create a new illustration\n"
    if n_characters > 0:
        keyboard_buttons.append(KeyboardButton(text=SHOW_CHARACTERS))
        _message += f"*{SHOW_CHARACTERS}* to see all characters\n" 
    keyboard_buttons.append(KeyboardButton(text="Help"))
    _message += f"Press *{HELP}* for help!\n" 
    
    
    keyboard = ReplyKeyboardMarkup(
        keyboard=[keyboard_buttons],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await message.reply(
        _message, reply_markup=keyboard, parse_mode="markdown")


@router.message(aif.text, Command(REDO))
async def send_help(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command {REDO}")
    await message.reply(f"{REDO} is not realised")


@router.message(aif.text, Command(HELP))
async def send_help(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command {HELP}")
    await message.reply(HELP_MESSAGE)
    
    
async def send_small_help(message: Message) -> None:
    await message.reply(SMALL_HELP_MESSAGE)


@router.message(aif.text, Command(CREATE_CHARACTER))
async def send_new_character(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command {NEW_CHARACTER}")
    if user_id not in users_state:
        users_state[user_id] = UserState()
    await message.answer("Describe your character with one word\n")
    users_state[user_id].state = NEW_CHARACTER
    users_state[user_id].character = None


async def processing_show_characters(n_characters, message: Message) -> None:
    if n_characters == 1:
        await message.reply(
            f"{n_characters} character is created. Here it is:"
        )
    else:
        await message.reply(
            f"{n_characters} characters are created. Here they are:"
        )
    created_characters = [crt for crt in characters if crt.model_path is not None]
    for crt in created_characters:
        await show_image(crt.target_image_path, crt.subject, message)
    keyboard_buttons = []
    for crt in created_characters:
        keyboard_buttons.append(KeyboardButton(text=crt.base_concept))
    keyboard_buttons.append(KeyboardButton(text="new character"))
    
    
    keyboard = ReplyKeyboardMarkup(
        keyboard=[keyboard_buttons],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await message.answer("Choose character", reply_markup=keyboard)



@router.message(aif.text, Command(SHOW_CHARACTERS))
async def send_show_characters(message: Message) -> None:

    user_id = message.from_user.id
    logger.info(f"User {user_id} command {SHOW_CHARACTERS}")
    if user_id not in users_state:
        users_state[user_id] = UserState()

    n_characters = get_n_characters()
    if n_characters == 0:
        users_state[user_id].state = NEW_CHARACTER
        await message.answer(
            "No character is created yet\nLet's create a new character\nMake a one word description of the character\n"
        )
        return
    else:
        users_state[user_id].state = UNDEFINED
        await processing_show_characters(n_characters, message)


@router.message(aif.text, Command("new_illustration"))
async def send_new_illustration(message: Message) -> None:
    user_id = message.from_user.id
    if user_id not in users_state:
        users_state[user_id] = UserState()
    logger.info(f"User {user_id} command new_illustration")
    if users_state[user_id].character is None:
        n_characters = get_n_characters()
        _str =  "You need to define your character first\n"
        _str += "Look at already created characters\n" if n_characters > 0 else "Describe your chracter in one word\n"
        await message.reply(
            _str
        )
        if n_characters > 0:
            await send_show_characters(message)
        else:
            users_state[user_id].state = NEW_CHARACTER
        return

    users_state[user_id].state = NEW_ILLUSTRATION
    await message.reply(
            "Describe what should be shown on the illustration\n"
        )    


async def show_image(image_path, description, message: Message):
    if not os.path.exists(image_path):
        await message.answer(f"🚫 File '{image_path}' not found.")
        return
        
    # Sending the image
    photo = FSInputFile(image_path)
    await bot.send_photo(message.chat.id, photo, caption=description)


async def processing_new_illustration(user_id, message: Message):
    exp_path = characters[users_state[user_id].character].exp_path
    model_path = characters[users_state[user_id].character].model_path
    base_concept = characters[users_state[user_id].character].base_concept
    subject = characters[users_state[user_id].character].subject
    
    user_message = message.text
    await message.reply(
        f"Creating an illustration:\n"
        f"{user_message} for character {subject}\n"
        f"Wait, it can take up to 60 seconds\n")

    inference(
        exp_path, model_path, subject, base_concept, [user_message])
    file_name = "_".join(user_message.lower().replace(",","").split(" "))
    image_path = f"/workspace/experiments/{exp_path}/{model_path}/inference/{file_name}_step_100.jpg"
    
    logger.info(f"file is created {file_name}")
    await show_image(image_path, base_concept, message)
    users_state[user_id].state = UNDEFINED

    keyboard_buttons = []
    keyboard_buttons.append(KeyboardButton(text=CREATE_ILLUSTRATION))        
    keyboard_buttons.append(KeyboardButton(text=SHOW_CHARACTERS))        
    keyboard_buttons.append(KeyboardButton(text=CREATE_CHARACTER))     
    keyboard_buttons.append(KeyboardButton(text=REDO))        
    keyboard_buttons.append(KeyboardButton(text=HELP))        
    keyboard = ReplyKeyboardMarkup(
        keyboard=[keyboard_buttons],
        resize_keyboard=True,
        one_time_keyboard=True
    )
    await message.answer("Choose action", reply_markup=keyboard)
        
    
async def create_target_image(user_id, message: Message):
    await message.reply(
        f"Image for character\n{characters[users_state[user_id].character].subject}\nis creating...\n"
        f"Wait, it can take up to 60 seconds\n"
    )
    
    id = str(uuid.uuid4())[:6]
    concept_token = [characters[users_state[user_id].character].base_concept]
    subject = characters[users_state[user_id].character].subject        
    exp_path = f"{concept_token[0]}.{id}"
    characters[users_state[user_id].character].exp_path = exp_path
    
    generate_target(exp_path, subject, concept_token)

    image_path = f"/workspace/experiments/{exp_path}/target.jpg"
    characters[users_state[user_id].character].target_image_path = image_path      
    logger.info(f"target image is created, {exp_path}")
    
    await show_image(
        image_path, f"{subject}\nthe image of the main character",
        message)


async def create_base_image(user_id, message: Message):
    await message.answer(
        "Additional images are created...\n"
        "Wait, it can take up to 3 min\n"
    )
    
    concept_token = [characters[users_state[user_id].character].base_concept]
    subject = characters[users_state[user_id].character].subject        
    exp_path = characters[users_state[user_id].character].exp_path        

    generate_base(exp_path, subject, concept_token)        
    image_path = f"/workspace/experiments/{exp_path}/base_all.jpg"
    characters[users_state[user_id].character].base_all_image_path = image_path      
    await show_image(
        image_path, f"{subject}\nthe additional images for the model",
    message)
    logger.info(f"base images are created, {exp_path}")


async def tune_model(user_id, message: Message):
    await message.answer(
        "Model is tunning...\n"
        "Wait, it can take up to 7 min\n"
    )
    
    concept_token = [characters[users_state[user_id].character].base_concept]
    subject = characters[users_state[user_id].character].subject        
    exp_path = characters[users_state[user_id].character].exp_path        

    tune(exp_path, 'model_Mask', subject, concept_token)
    
    logger.info(f"model is tuned, {exp_path}")
    characters[users_state[user_id].character].model_path = 'model_Mask'


async def processing_new_character(user_id, message: Message):
    if users_state[user_id].character is None:
        if len(message.text.strip().split(" ")) > 1:
            await message.reply(
                "Make a one word description\n"
            )
        else:
            characters.append(CharacterStore(message.text.lower()))                                
            users_state[user_id].character = len(characters) - 1
            await message.reply(
                "Make a full description of the character\n"
            )
        return
    characters[users_state[user_id].character].subject = message.text.lower()
    await create_target_image(user_id, message)
    await create_base_image(user_id, message)
    await tune_model(user_id, message)

    # save_characters()
    
    await message.answer(
        "Model is ready!\n"
        "Now you can create your illustration\n"
        "Describe what should be shown on the illustration\n"
    )
    users_state[user_id].state = NEW_ILLUSTRATION




@router.message(aif.text)
async def processing_messages(message: Message):
    user_id = message.from_user.id
    logger.info(f"User {user_id} write {message.text}")

    if user_id not in users_state:
        users_state[user_id] = UserState()


    users_state[user_id].user_messages.append(message.text)
                
    if users_state[user_id].state == NEW_ILLUSTRATION:
        await processing_new_illustration(user_id, message)
        return


    if users_state[user_id].state == NEW_CHARACTER:
        await processing_new_character(user_id, message)
        return            

    message_is_base_concept = False
    for n, crt in enumerate(characters):
        if message.text == crt.base_concept:
            message_is_base_concept = True
            users_state[user_id].character = n
            logger.info(f"User {user_id} chose character {n}.")
            break
 
    if message_is_base_concept:
        users_state[user_id].state = NEW_ILLUSTRATION
        await message.reply(
            f"Character {message.text} is selected\n"
            "Describe what should be shown on the illustration\n"
        )
        return            
        
    if message.text == CREATE_ILLUSTRATION:
        await send_new_illustration(message)
        return
    if message.text == SHOW_CHARACTERS:
        await send_show_characters(message)
        return
    if message.text == "Change character":
        await send_show_characters(message)
        return
    if message.text == CREATE_CHARACTER:
        await send_new_character(message)
        return
    if message.text == HELP:
        await send_help(message)
        return
    if message.text == REDO:
        await send_help(message)
        return
    logger.info(f"User {user_id} state is {users_state[user_id].state} and can't be processed.")
    await send_welcome(message)       
    return            
    
                

async def main() -> None:
    # load_caracters()
    
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)
    logger.info(f"Bot is starting")

    
if __name__ == "__main__":
    asyncio.run(main())
