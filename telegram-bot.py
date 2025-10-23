import os
import logging
from datetime import datetime
from tempfile import gettempdir
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import faster_whisper
from pydub import AudioSegment
import math

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

def split_audio_into_segments(audio_path, segment_duration=30, overlap=2):
    """
    Разбивает аудиофайл на сегменты заданной длительности с перекрытием
    
    Args:
        audio_path: путь к аудиофайлу
        segment_duration: длительность сегмента в секундах (по умолчанию 30)
        overlap: перекрытие между сегментами в секундах (по умолчанию 2)
    
    Returns:
        list: список путей к сегментам
    """
    try:
        logger.info(f"Начало разбиения аудио: {audio_path}")
        
        # Загружаем аудиофайл
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        duration_seconds = duration_ms / 1000
        
        logger.info(f"Длительность аудио: {duration_seconds:.2f} секунд")
        
        # Если файл короче 30 секунд, возвращаем как есть
        if duration_seconds <= segment_duration:
            logger.info("Аудио короче 30 секунд, фрагментация не нужна")
            return [audio_path]
        
        # Создаем директорию для сегментов
        base_dir = os.path.dirname(audio_path)
        segments_dir = os.path.join(base_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        
        segment_paths = []
        segment_duration_ms = segment_duration * 1000
        overlap_ms = overlap * 1000
        
        # Вычисляем количество сегментов
        num_segments = math.ceil((duration_ms - overlap_ms) / (segment_duration_ms - overlap_ms))
        logger.info(f"Будет создано {num_segments} сегментов")
        
        for i in range(num_segments):
            start_time = i * (segment_duration_ms - overlap_ms)
            end_time = min(start_time + segment_duration_ms, duration_ms)
            
            # Извлекаем сегмент
            segment = audio[start_time:end_time]
            
            # Сохраняем сегмент
            segment_filename = f"segment_{i+1:03d}.wav"
            segment_path = os.path.join(segments_dir, segment_filename)
            segment.export(segment_path, format="wav")
            
            segment_paths.append(segment_path)
            logger.debug(f"Создан сегмент {i+1}/{num_segments}: {segment_filename} ({len(segment)/1000:.2f}с)")
        
        logger.info(f"Создано {len(segment_paths)} сегментов")
        return segment_paths
        
    except Exception as e:
        logger.error(f"Ошибка разбиения аудио: {str(e)}", exc_info=True)
        raise RuntimeError("Ошибка разбиения аудио на сегменты") from e

def cleanup_segments(segment_paths):
    """Удаляет временные сегменты"""
    try:
        for segment_path in segment_paths:
            if os.path.exists(segment_path):
                os.remove(segment_path)
                logger.debug(f"Удален сегмент: {segment_path}")
        
        # Удаляем директорию сегментов
        if segment_paths:
            segments_dir = os.path.dirname(segment_paths[0])
            if os.path.exists(segments_dir):
                os.rmdir(segments_dir)
                logger.debug(f"Удалена директория сегментов: {segments_dir}")
                
    except Exception as e:
        logger.warning(f"Ошибка очистки сегментов: {str(e)}")

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

        # Проверяем размер файла и разбиваем на сегменты если нужно
        file_size = os.path.getsize(audio_path)
        logger.info(f"Размер файла: {file_size / (1024*1024):.2f} МБ")
        
        # Разбиваем аудио на сегменты
        try:
            segment_paths = split_audio_into_segments(audio_path)
            logger.info(f"Создано {len(segment_paths)} сегментов для обработки")
        except Exception as e:
            await update.message.reply_text('❌ Ошибка разбиения аудио')
            raise
        
        # Транскрибация всех сегментов
        all_texts = []
        try:
            for i, segment_path in enumerate(segment_paths):
                logger.info(f"Обработка сегмента {i+1}/{len(segment_paths)}")
                
                # Обновляем статус для пользователя
                if len(segment_paths) > 1:
                    await update.message.reply_text(f'⏳ Обрабатываю сегмент {i+1}/{len(segment_paths)}...')
                
                try:
                    segment_text = transcriber.transcribe(segment_path)
                    if segment_text.strip():
                        all_texts.append(segment_text.strip())
                        logger.info(f"Сегмент {i+1} обработан: {len(segment_text)} символов")
                    else:
                        logger.warning(f"Сегмент {i+1} не содержит текста")
                except Exception as e:
                    logger.error(f"Ошибка обработки сегмента {i+1}: {str(e)}")
                    # Продолжаем обработку других сегментов
                    continue
            
            # Объединяем все результаты
            if all_texts:
                text = ' '.join(all_texts)
                logger.info(f"Успешная транскрибация всех сегментов. Общая длина текста: {len(text)} символов")
            else:
                text = ""
                logger.warning("Не удалось обработать ни одного сегмента")
                
        except Exception as e:
            await update.message.reply_text('❌ Ошибка распознавания')
            raise
        finally:
            # Очищаем все временные файлы
            os.remove(audio_path)
            logger.info("Основной аудиофайл удален")
            
            # Удаляем сегменты
            cleanup_segments(segment_paths)

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
