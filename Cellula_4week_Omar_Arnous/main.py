import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction
from src.rag import answer_question, clear_history

import logging
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")
telegram_token = os.getenv("TELEGRAM")

LOG_DIR = "logs"  # or "../log" if you want outside the src folder
os.makedirs(LOG_DIR, exist_ok=True)

# Simple reply logic
def get_reply(user_input: str, chat_id) -> str:
    logger.info(f"User ({chat_id}) said: {user_input}")
    user_input = user_input.lower()

    if user_input == "cls":
        clear_history(chat_id)
        return "Chat history cleared."

    answer, similar_docs = answer_question(user_input, chat_id)
    # save similar_docs to a log file for debugging
    log_file_path = os.path.join(LOG_DIR, f"retrieved_docs_log({chat_id}).txt")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\nUser ({chat_id}) question: {user_input}\n")
        for i, doc in enumerate(similar_docs):
            f.write(f"\nDocument {i+1}:\n{doc.page_content}\nTitle: {doc.metadata.get('title', 'N/A')}\nSource: {doc.metadata.get('source', 'N/A')}\n{'-'*50}\n")
        f.write(f"Answer:\n{answer}\n{'='*100}\n\n")
    
    return answer if answer else "Sorry, I couldn't find an answer to your question."
    
# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I'm your Customer Support AI Bot 🤖.\n"
        "Ask me anything about our products, services, or support process!")

# Message handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    reply = get_reply(user_input, chat_id)
    await update.message.reply_text(reply)

async def error_handler(update, context):
    logger.error(f"Exception while handling an update: {context.error}")

logger.info("Loading bot...")
if __name__ == '__main__':
    logger.info("Starting bot...")
    app = ApplicationBuilder().token(telegram_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot is waiting for messages...")
    app.run_polling()