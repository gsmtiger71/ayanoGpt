import asyncio
import logging
import sys
import requests
import json
import os

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# ================== CONFIG ==================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
LONGCAT_API_KEY = os.getenv("LONGCAT_API_KEY", "")
LONGCAT_API_URL = "https://api.longcat.chat/openai/v1/chat/completions"

BOT_CREDIT = "\n\n━━━━━━━━━━━━━━━━━━\nBot Developed by - @DevloperAyano"

USER_DATA_FILE = "user_data.json"
CONVERSATIONS_FILE = "conversations.json"

# user_id -> list[{"role": "...", "content": "..."}]
user_conversations: dict[int, list[dict]] = {}
# set of user_ids who accepted terms
user_accepted_terms: set[int] = set()


class BotStates(StatesGroup):
    waiting_for_acceptance = State()
    chatting = State()


logging.basicConfig(level=logging.INFO, stream=sys.stdout)


# ================== PERSISTENCE ==================

def load_user_data() -> None:
    """Load accepted users and conversations from JSON files."""
    global user_accepted_terms, user_conversations

    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                user_accepted_terms = set(data.get("accepted_users", []))
                logging.info(f"Loaded {len(user_accepted_terms)} users who accepted terms")

        if os.path.exists(CONVERSATIONS_FILE):
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                conv = json.load(f)
                user_conversations = {int(k): v for k, v in conv.items()}
                logging.info(f"Loaded conversations for {len(user_conversations)} users")

    except Exception as e:
        logging.error(f"Error loading user data: {e}")


def save_user_data() -> None:
    """Save accepted users and conversations to JSON files."""
    try:
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"accepted_users": list(user_accepted_terms)}, f, ensure_ascii=False, indent=2)

        with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_conversations, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logging.error(f"Error saving user data: {e}")


def add_accepted_user(user_id: int) -> None:
    """Mark user as accepted and persist."""
    user_accepted_terms.add(user_id)
    save_user_data()


# ================== HELPERS ==================

def clean_markdown(text: str) -> str:
    """Remove basic Markdown markers so answer looks clean."""
    # CRITICAL: always 2 arguments to replace()
    text = text.replace("**", "")
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("```", "")
    text = text.replace("_", "")
    text = text.replace("~~", "")
    return text.strip()


def create_terms_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Accept Terms & Continue", callback_data="accept_terms")],
            [InlineKeyboardButton(text="❌ Decline", callback_data="decline_terms")],
        ]
    )


# ================== LONGCAT API ==================

async def get_ai_response(user_message: str, user_id: int) -> str:
    """Call LongCat API and return answer or user-friendly error."""

    if user_id not in user_conversations:
        user_conversations[user_id] = []

    user_conversations[user_id].append({"role": "user", "content": user_message})

    # Keep last 10 messages only
    if len(user_conversations[user_id]) > 10:
        user_conversations[user_id] = user_conversations[user_id][-10:]

    headers = {
        "Authorization": f"Bearer {LONGCAT_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "LongCat-Flash-Chat",
        "messages": user_conversations[user_id],
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    try:
        response = await asyncio.to_thread(
            requests.post,
            LONGCAT_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        try:
            resp_json = response.json()
        except Exception as e:
            logging.error(f"JSON parse error: {e}, text={response.text[:400]}")
            return "❌ API returned invalid JSON. Please try again."

        if response.status_code == 429:
            return "⚠️ Rate limit exceeded. Please try again in a minute."

        if response.status_code != 200:
            if isinstance(resp_json, dict) and "error" in resp_json:
                err = resp_json["error"]
                msg = err.get("message", "Unknown API error")
                return f"❌ Error: {msg}"
            logging.error(f"Unexpected non-200 response: {response.status_code}, json={resp_json}")
            return f"❌ Error: HTTP {response.status_code}"

        # Expected OpenAI-style:
        # {"choices": [ { "index": 0, "message": { "role": "assistant", "content": "..." }, ... } ]}
        if isinstance(resp_json, dict) and "choices" in resp_json:
            choices = resp_json.get("choices", [])
            if not isinstance(choices, list) or not choices:
                logging.error(f"Invalid choices field: {choices}")
                return "❌ API returned empty or invalid choices."

            raw_first = choices[0]

            # Your logged structure shows first_choice is a list wrapping the dict:
            # [{'index': 0, 'message': {...}}]
            # So handle both list and dict cases
            first_choice = raw_first
            if isinstance(raw_first, list):
                if not raw_first:
                    logging.error(f"first_choice list is empty: {raw_first}")
                    return "❌ API returned invalid choice structure."
                first_choice = raw_first[0]

            if not isinstance(first_choice, dict):
                logging.error(f"Invalid choice structure: {raw_first}")
                return "❌ API returned invalid choice structure."

            msg_obj = first_choice.get("message") or first_choice.get("delta") or {}
            if not isinstance(msg_obj, dict):
                logging.error(f"Invalid message object: {msg_obj}")
                return "❌ API returned invalid message object."

            content = msg_obj.get("content")
            if not content:
                logging.error(f"No content field in message: {resp_json}")
                return "❌ API did not return any content."

            user_conversations[user_id].append({"role": "assistant", "content": content})
            save_user_data()
            return content

        logging.error(f"Completely unexpected JSON structure: {type(resp_json)} -> {resp_json}")
        return "❌ Unexpected API response format. Please try again."

    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except Exception as e:
        logging.error(f"Exception in get_ai_response: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"


# ================== HANDLERS ==================

async def start_handler(message: Message, state: FSMContext) -> None:
    """Handle /start: welcome screen or quick welcome-back."""
    user_id = message.from_user.id
    user_name = message.from_user.full_name or "there"

    if user_id in user_accepted_terms:
        txt = (
            f"👋 <b>Welcome back, {html.quote(user_name)}!</b>\n\n"
            "✅ You are already set up.\n\n"
            "💬 Just send your question as a normal message."
        )
        await message.answer(txt, parse_mode=ParseMode.HTML)
        await state.set_state(BotStates.chatting)
        return

    welcome_text = f"""╔═══════════════════════════════╗
║  🤖 <b>WELCOME TO LONGCAT AI</b> 🤖  ║
╚═══════════════════════════════╝

👋 <b>Hello {html.quote(user_name)}!</b>

🌟 <i>This bot uses LongCat-Flash-Chat via LongCat API.</i>

━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 What I Can Do:</b>
• Answer your questions
• Explain concepts and code
• Help with study and research
• Chat in natural language

━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🚀 How To Use:</b>
Just send messages like you chat normally.
No commands needed after setup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📋 Terms:</b>
• Uses LongCat free API quota
• No illegal or abusive content
• Developer not responsible for AI output

━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>⚡ Tap “Accept” to start.</b>"""

    await message.answer(
        welcome_text,
        reply_markup=create_terms_keyboard(),
        parse_mode=ParseMode.HTML,
    )
    await state.set_state(BotStates.waiting_for_acceptance)


async def accept_terms_handler(callback: CallbackQuery, state: FSMContext) -> None:
    user_id = callback.from_user.id
    add_accepted_user(user_id)

    txt = (
        "✅ <b>Terms Accepted!</b>\n\n"
        "🎉 You can now start chatting.\n\n"
        "Just send your questions as normal messages."
    )
    await callback.message.edit_text(txt, parse_mode=ParseMode.HTML)
    await callback.answer("Welcome!")
    await state.set_state(BotStates.chatting)


async def decline_terms_handler(callback: CallbackQuery, state: FSMContext) -> None:
    txt = (
        "❌ <b>Terms Declined</b>\n\n"
        "To use this bot, send /start again and accept the terms."
    )
    await callback.message.edit_text(txt, parse_mode=ParseMode.HTML)
    await callback.answer("Terms declined")
    await state.clear()


async def message_handler(message: Message, state: FSMContext) -> None:
    """Handle all text messages after terms acceptance."""
    user_id = message.from_user.id

    if user_id not in user_accepted_terms:
        await message.answer(
            "⚠️ <b>Please send /start and accept the terms first.</b>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Ensure returning users have chatting state even after restart
    if await state.get_state() is None:
        await state.set_state(BotStates.chatting)
        logging.info(f"Auto-set chatting state for user {user_id}")

    user_message = message.text
    await message.chat.do("typing")

    ai_response = await get_ai_response(user_message, user_id)
    cleaned = clean_markdown(ai_response)
    formatted = f"<blockquote>{html.quote(cleaned)}</blockquote>{BOT_CREDIT}"

    try:
        await message.answer(formatted, parse_mode=ParseMode.HTML)
    except Exception as e:
        logging.error(f"Error sending formatted message: {e}")
        await message.answer(f"{html.quote(cleaned)}{BOT_CREDIT}", parse_mode=ParseMode.HTML)


# ================== MAIN ==================

async def main() -> None:
    logging.info("Loading user data...")
    load_user_data()

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    dp.message.register(start_handler, CommandStart())
    dp.callback_query.register(accept_terms_handler, F.data == "accept_terms")
    dp.callback_query.register(decline_terms_handler, F.data == "decline_terms")
    dp.message.register(message_handler, F.text)

    logging.info("==============================================")
    logging.info("LongCat Telegram Bot started.")
    logging.info(f"Accepted users: {len(user_accepted_terms)}")
    logging.info(f"Users with conversation history: {len(user_conversations)}")
    logging.info("==============================================")

    try:
        await dp.start_polling(bot)
    finally:
        logging.info("Saving user data before shutdown...")
        save_user_data()
        logging.info("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        save_user_data()

