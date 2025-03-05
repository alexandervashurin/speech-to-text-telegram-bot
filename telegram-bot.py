import os
import logging
from datetime import datetime
from tempfile import gettempdir
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import faster_whisper

# Настройка логгера
logging.basicConfig(
    level=logging.DEBUG,  # Включаем детальное логирование
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

class Transcriber:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="int8"):
        logger.info(f"Инициализация модели Whisper {model_size}")
        try:
            self.model = faster_whisper.WhisperModel(
                model_size, 
                device=device, 
                compute_type=compute_type
            )
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def transcribe(self, file_path):
        """Транскрибация аудиофайла с улучшенным логированием"""
        try:
            logger.info(f"Начало транскрибации файла: {file_path}")
            segments, info = self.model.transcribe(
                file_path,
                beam_size=5,
                vad_filter=True,
                language="ru"
            )
            
            text = ' '.join(segment.text for segment in segments).strip()
            logger.debug(f"Сырой текст транскрибации: {text[:500]}...")  # Логируем первые 500 символов
            return text
        
        except Exception as e:
            logger.error(f"Ошибка транскрибации: {str(e)}", exc_info=True)
            raise RuntimeError("Ошибка распознавания аудио") from e

# Инициализация транскрайбера
try:
    transcriber = Transcriber()
except Exception as e:
    logger.critical(f"Невозможно инициализировать модель: {str(e)}")
    exit(1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('🎙 Привет! Отправь мне голосовое сообщение или аудиофайл.')

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    logger.info(f"Новый запрос от {user.id} ({user.username})")
    
    try:
        audio_file = update.message.audio or update.message.voice
        if not audio_file:
            await update.message.reply_text('❌ Отправьте аудиофайл.')
            return

        # Создание временной директории
        temp_dir = os.path.join(gettempdir(), f"tg_audio_{user.id}")
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Временная директория: {temp_dir}")

        # Скачивание файла
        await update.message.reply_text('⏳ Обрабатываю аудио...')
        file = await context.bot.get_file(audio_file.file_id)
        audio_path = os.path.join(temp_dir, f"audio_{audio_file.file_id}.ogg")
        await file.download_to_drive(audio_path)
        logger.info(f"Аудио сохранено: {audio_path} ({os.path.getsize(audio_path)} байт)")

        # Транскрибация
        try:
            text = transcriber.transcribe(audio_path)
            logger.info(f"Успешная транскрибация. Длина текста: {len(text)} символов")
        except Exception as e:
            await update.message.reply_text('❌ Ошибка распознавания')
            raise
        finally:
            os.remove(audio_path)
            logger.info("Аудиофайл удален")

        if not text.strip():
            await update.message.reply_text('❌ Не удалось распознать речь')
            return

        # Отправка результата
        if len(text) <= 4096:
            await update.message.reply_text(f"📝 Результат:\n\n{text}")
            logger.info("Текст отправлен сообщением")
        else:
            filename = f"Транскрипция_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
            text_path = os.path.join(temp_dir, filename)
            
            try:
                # Запись текста с проверкой
                logger.info(f"Начало записи в {filename}")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    logger.info(f"Записано {len(text)} символов")

                # Верификация записи
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)
                    logger.debug(f"Проверка содержимого:\n{content}...")
                
                # Отправка файла
                await update.message.reply_document(
                    document=InputFile(text_path, filename=filename),
                    caption="📁 Текст слишком длинный"
                )
                logger.info("Файл отправлен")
            
            except Exception as e:
                logger.error(f"Ошибка записи: {str(e)}", exc_info=True)
                await update.message.reply_text('⚠️ Ошибка создания файла')
            
            finally:
                if os.path.exists(text_path):
                    os.remove(text_path)
                    logger.info("Текстовый файл удален")

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
        await update.message.reply_text('⚠️ Внутренняя ошибка сервера')

def main():
    try:
        logger.info("🚀 Запуск бота...")
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
        app.run_polling()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
