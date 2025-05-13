import asyncio
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import logging
from aiogram import Bot, Dispatcher
from aiogram import F as aif

from aiogram.dispatcher.router import Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove, FSInputFile
from aiogram.utils.formatting import Text
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.dispatcher.router import Router
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.fsm.storage.memory import MemoryStorage

import uuid

API_TOKEN = '7552786080:AAEiKMsAUkoV3U84w02e9ZHLHxtK9pGqvuI'


from generate_data_target import generate_target
from generate_data_base import generate_base
from tune_mask import tune
from inference_mask import inference

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
    /history – message history
    /show_image – show sample image
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
NEW_CHARACTOR = 1
NEW_ILLUSTRATION = 2

class CharacterStore:
    def __init__(self, base_concept):
        self.base_concept = base_concept.lower()
        self.subject = None
        self.target_image_path = None
        self.exp_path = None
        self.model_path = None

class UserState:
    def __init__(self):
        self.state = None
        self.character = None
        self.user_messages = []


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


@router.message(aif.text, Command("start"))
async def send_welcome(message: Message) -> None:
    logger.info(f"User {message.from_user.id} started the bot")
    await message.reply(
        "Welcome! Let's create illustations for your story.\n"
        "/help for help!\n", reply_markup=ReplyKeyboardRemove()
    )


@router.message(aif.text, Command("help"))
async def send_help(message: Message) -> None:
    await message.reply(HELP_MESSAGE)
    
    
async def send_small_help(message: Message) -> None:
    await message.reply(SMALL_HELP_MESSAGE)


@router.message(aif.text, Command("new_character"))
async def send_new_character(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command new_character")
    if user_id not in users_state:
        users_state[user_id] = UserState()
    await message.reply("Describe your character with one word\n")
    users_state[user_id].state = NEW_CHARACTOR
    users_state[user_id].character = None


@router.message(aif.text, Command("new_illustration"))
async def send_new_illustration(message: Message) -> None:
    user_id = message.from_user.id
    if user_id not in users_state:
        users_state[user_id] = UserState()
    logger.info(f"User {user_id} command new_illustration")
    if users_state[user_id].character is None:
        await message.reply(
            "You need to define your character first\n"
        )
        await send_new_character(message)
        return

    await message.reply(
            "Describe what should be shown on the illustration\n"
        )    
    users_state[user_id].state = NEW_ILLUSTRATION


async def show_image(image_path, base_concept, message: Message):
    if not os.path.exists(image_path):
        await message.reply(f"🚫 File '{image_path}' not found.")
        return
        
    # Sending the image
    photo = FSInputFile(image_path)
    await bot.send_photo(message.chat.id, photo, caption=f"Here is an image of {base_concept}")


async def display(crt, message: Message):
    await show_image(crt.image_path, crt.base_concept, message)


@router.message(aif.text, Command("show_characters"))
async def send_show_characters(message: Message) -> None:
    user_id = message.from_user.id
    logger.info(f"User {user_id} command show_characters")

    if user_id not in users_state:
        users_state[user_id] = UserState()

    created_characters = [crt for crt in characters if crt.image_path is not None]
    if len(created_characters) == 0:
        await message.reply(
            "No character is created jet\nLet's create new character"
        )
        await send_new_character(message)
    else:
        await message.reply(
            f"{len(created_characters)} characters are created. Here are they:"
        )        
        for crt in characters:
            if crt.image_path is not None:
                await display(crt, message)
        await send_small_help(message)



async def wait_for_file(file_path, check_interval=2, timeout=5):
    logger.info(f"Wait for file  {file_path}")

    start_time = asyncio.get_event_loop().time()
    while not os.path.exists(file_path):
        if asyncio.get_event_loop().time() - start_time > timeout:
            return False
        await asyncio.sleep(check_interval)
    return True


@router.message(aif.text)
async def processing_messages(message: Message):
    user_id = message.from_user.id
    user_message = message.text
    logger.info(f"User {user_id} write {message.text}")

    if user_id not in users_state:
        users_state[user_id] = UserState()


    users_state[user_id].user_messages.append(user_message)
                
    if users_state[user_id].state == NEW_ILLUSTRATION:
        await message.reply(f"""Creating new illustration:
                            {user_message}
                            Wait, it can take up to 60 seconds\n""")

        exp_path = characters[users_state[user_id].character].exp_path
        model_path = characters[users_state[user_id].character].model_path
        base_concept = characters[users_state[user_id].character].base_concept
        

        inference(
            exp_path, model_path,
            characters[users_state[user_id].character].subject,
            base_concept, [user_message])
        file_name = "_".join(user_message.lower().replace(",","").split(" "))
        image_path = f"/workspace/experiments/{exp_path}/{model_path}/inference/{file_name}_step_100.jpg"
        
        logger.info(f"file is created {file_name}")
        await show_image(image_path, base_concept, message)
        users_state[user_id].state = UNDEFINED
    
        return


    if users_state[user_id].state == NEW_CHARACTOR:
        if users_state[user_id].character is None:
            if len(user_message.strip().split(" ")) > 1:
                await message.reply(
                    "Make a one word description\n"
                )
            else:
                characters.append(CharacterStore(user_message.lower()))
                                  
                users_state[user_id].character = len(characters) - 1
                await message.reply(
                    "Make a full description of the character\n"
                )
            return
        characters[users_state[user_id].character].subject = user_message
        
        await message.reply(
            "Character image is creating\n"
            "Wait, it can take up to 60 seconds\n"
        )
        
        id = str(uuid.uuid4())[:6]
        concept_token = [characters[users_state[user_id].character].base_concept]
        subject = characters[users_state[user_id].character].subject        
        exp_path = f"{concept_token[0]}.{id}"
        characters[users_state[user_id].character].exp_path = exp_path
        
        generate_target(exp_path, subject, concept_token)

        image_path = f"/workspace/experiments/{exp_path}/target.jpg"
        characters[users_state[user_id].character].image_path = image_path      
        logger.info(f"target image is created, {exp_path}")
        
        await display(characters[users_state[user_id].character], message)
        await message.reply(
            "Additional images are created...\n"
            "Wait, it can take up to 3 min\n"
        )
        generate_base(exp_path, subject, concept_token)        
        logger.info(f"base images are created, {exp_path}")

 
        tune(exp_path, 'model_Mask', subject, concept_token)
        characters[users_state[user_id].character].model_path = 'model_Mask'
        users_state[user_id].state = UNDEFINED
        await send_small_help(message)
        return            

    logger.info(f"User {user_id} state is {users_state[user_id].state} and can't be processed.")
                
                

async def main() -> None:
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)
    logger.info(f"Bot is starting")

    
if __name__ == "__main__":
    asyncio.run(main())