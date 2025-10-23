import os
import logging
import shutil
from datetime import datetime
from tempfile import gettempdir
from typing import List, Optional, Tuple, Dict, Any
from telegram import Update, InputFile
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from dotenv import load_dotenv
from pydub import AudioSegment
import speech_recognition as sr
import math

# Для Python 3.7 совместимости
try:
    from dataclasses import dataclass
except ImportError:
    # Fallback для старых версий Python
    def dataclass(cls):
        return cls

# Настройка логгера
logging.basicConfig(
    level=logging.DEBUG,  # Включаем детальное логирование
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class BotConfig:
    """Конфигурация бота для Python 3.7"""
    
    def __init__(self, telegram_token: str, model_size: str = "large-v3", 
                 device: str = "cpu", compute_type: str = "int8",
                 segment_duration: int = 30, segment_overlap: int = 2,
                 max_file_size_mb: int = 50, max_text_length: int = 4096,
                 beam_size: int = 5, language: str = "ru", vad_filter: bool = True):
        self.telegram_token = telegram_token
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.max_file_size_mb = max_file_size_mb
        self.max_text_length = max_text_length
        self.beam_size = beam_size
        self.language = language
        self.vad_filter = vad_filter

    @classmethod
    def from_env(cls):
        """Создает конфигурацию из переменных окружения"""
        token = os.getenv("TELEGRAM_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_TOKEN не найден в переменных окружения")
        
        return cls(
            telegram_token=token,
            model_size=os.getenv("MODEL_SIZE", "large-v3"),
            device=os.getenv("DEVICE", "cpu"),
            compute_type=os.getenv("COMPUTE_TYPE", "int8"),
            segment_duration=int(os.getenv("SEGMENT_DURATION", "30")),
            segment_overlap=int(os.getenv("SEGMENT_OVERLAP", "2")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "50")),
            max_text_length=int(os.getenv("MAX_TEXT_LENGTH", "4096")),
            beam_size=int(os.getenv("BEAM_SIZE", "5")),
            language=os.getenv("LANGUAGE", "ru"),
            vad_filter=os.getenv("VAD_FILTER", "true").lower() == "true"
        )

# Инициализация конфигурации
try:
    config = BotConfig.from_env()
except ValueError as e:
    logging.critical("Ошибка конфигурации: {}".format(e))
    exit(1)

class Transcriber:
    """Класс для транскрибации аудио с использованием SpeechRecognition"""
    
    def __init__(self, config):
        self.config = config
        self.recognizer = sr.Recognizer()
        logger.info("Инициализация SpeechRecognition")
        logger.info("Поддерживаемые языки: ru, en, es, fr, de, it, pt, pl, tr, uk")

    def transcribe(self, file_path):
        """Реальная транскрибация аудиофайла"""
        try:
            logger.info("Начало транскрибации файла: {}".format(file_path))
            
            # Конвертируем OGG в WAV для SpeechRecognition
            audio = AudioSegment.from_file(file_path)
            wav_path = file_path.replace('.ogg', '.wav')
            audio.export(wav_path, format="wav")
            logger.debug("Аудио конвертировано в WAV: {}".format(wav_path))
            
            # Загружаем WAV файл
            with sr.AudioFile(wav_path) as source:
                # Убираем фоновый шум
                self.recognizer.adjust_for_ambient_noise(source)
                # Записываем аудио
                audio_data = self.recognizer.record(source)
            
            # Распознаем речь
            try:
                # Пробуем Google Speech Recognition (бесплатно)
                text = self.recognizer.recognize_google(audio_data, language=self.config.language)
                logger.info("Успешное распознавание через Google Speech Recognition")
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition не смог распознать аудио")
                text = "Не удалось распознать речь. Попробуйте говорить четче."
            except sr.RequestError as e:
                logger.error("Ошибка запроса к Google Speech Recognition: {}".format(e))
                text = "Ошибка подключения к сервису распознавания речи."
            
            # Удаляем временный WAV файл
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logger.debug("Временный WAV файл удален")
            
            logger.debug("Результат транскрибации: {}...".format(text[:500]))
            return text
        
        except Exception as e:
            logger.error("Ошибка транскрибации: {}".format(str(e)), exc_info=True)
            raise RuntimeError("Ошибка распознавания аудио") from e

class AudioProcessor:
    """Класс для обработки аудиофайлов"""
    
    def __init__(self, config):
        self.config = config
    
    def split_audio_into_segments(self, audio_path):
        """
        Разбивает аудиофайл на сегменты заданной длительности с перекрытием
        
        Args:
            audio_path: путь к аудиофайлу
        
        Returns:
            список путей к сегментам
        """
        try:
            logger.info("Начало разбиения аудио: {}".format(audio_path))
            
            # Загружаем аудиофайл
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            duration_seconds = duration_ms / 1000
            
            logger.info("Длительность аудио: {:.2f} секунд".format(duration_seconds))
            
            # Если файл короче заданной длительности, возвращаем как есть
            if duration_seconds <= self.config.segment_duration:
                logger.info("Аудио короче {} секунд, фрагментация не нужна".format(self.config.segment_duration))
                return [audio_path]
            
            # Создаем директорию для сегментов
            base_dir = os.path.dirname(audio_path)
            segments_dir = os.path.join(base_dir, "segments")
            os.makedirs(segments_dir, exist_ok=True)
            
            segment_paths = []
            segment_duration_ms = self.config.segment_duration * 1000
            overlap_ms = self.config.segment_overlap * 1000
            
            # Вычисляем количество сегментов
            num_segments = math.ceil((duration_ms - overlap_ms) / (segment_duration_ms - overlap_ms))
            logger.info("Будет создано {} сегментов".format(num_segments))
            
            for i in range(num_segments):
                start_time = i * (segment_duration_ms - overlap_ms)
                end_time = min(start_time + segment_duration_ms, duration_ms)
                
                # Извлекаем сегмент
                segment = audio[start_time:end_time]
                
                # Сохраняем сегмент
                segment_filename = "segment_{:03d}.wav".format(i+1)
                segment_path = os.path.join(segments_dir, segment_filename)
                segment.export(segment_path, format="wav")
                
                segment_paths.append(segment_path)
                logger.debug("Создан сегмент {}/{}: {} ({:.2f}с)".format(
                    i+1, num_segments, segment_filename, len(segment)/1000))
            
            logger.info("Создано {} сегментов".format(len(segment_paths)))
            return segment_paths
            
        except Exception as e:
            logger.error("Ошибка разбиения аудио: {}".format(str(e)), exc_info=True)
            raise RuntimeError("Ошибка разбиения аудио на сегменты") from e

    def cleanup_segments(self, segment_paths):
        """Удаляет временные сегменты"""
        try:
            for segment_path in segment_paths:
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                    logger.debug("Удален сегмент: {}".format(segment_path))
            
            # Удаляем директорию сегментов
            if segment_paths:
                segments_dir = os.path.dirname(segment_paths[0])
                if os.path.exists(segments_dir):
                    os.rmdir(segments_dir)
                    logger.debug("Удалена директория сегментов: {}".format(segments_dir))
                    
        except Exception as e:
            logger.warning("Ошибка очистки сегментов: {}".format(str(e)))
    
    def validate_audio_file(self, file_path):
        """Проверяет валидность аудиофайла"""
        try:
            if not os.path.exists(file_path):
                return False, "Файл не найден"
            
            file_size = os.path.getsize(file_path)
            max_size = self.config.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size:
                return False, "Файл слишком большой: {:.2f} МБ (максимум {} МБ)".format(
                    file_size / (1024*1024), self.config.max_file_size_mb)
            
            # Проверяем, что файл можно загрузить
            AudioSegment.from_file(file_path)
            return True, "Файл валиден"
            
        except Exception as e:
            return False, "Ошибка валидации: {}".format(str(e))

class TelegramBot:
    """Основной класс Telegram бота"""
    
    def __init__(self, config):
        self.config = config
        self.transcriber = Transcriber(config)
        self.audio_processor = AudioProcessor(config)
        self.app = None
    
    def start_command(self, update, context):
        """Обработчик команды /start"""
        update.message.reply_text('🎙 Привет! Отправь мне голосовое сообщение или аудиофайл.')
    
    def handle_audio(self, update, context):
        """Обработчик аудиосообщений"""
        user = update.message.from_user
        logger.info("Новый запрос от {} ({})".format(user.id, user.username))
        
        temp_dir = None
        segment_paths = []
        
        try:
            audio_file = update.message.audio or update.message.voice
            if not audio_file:
                update.message.reply_text('❌ Отправьте аудиофайл.')
                return

            # Проверяем размер файла (лимит Telegram: 50 МБ)
            if audio_file.file_size and audio_file.file_size > 50 * 1024 * 1024:  # 50 МБ
                update.message.reply_text(
                    "❌ Файл слишком большой ({} МБ). Максимальный размер: 50 МБ".format(
                        round(audio_file.file_size / (1024 * 1024), 1)
                    )
                )
                return
            
            # Проверяем длительность (лимит: 2 часа)
            if hasattr(audio_file, 'duration') and audio_file.duration and audio_file.duration > 7200:  # 2 часа
                update.message.reply_text(
                    "❌ Аудио слишком длинное ({} мин). Максимальная длительность: 2 часа".format(
                        round(audio_file.duration / 60, 1)
                    )
                )
                return

            # Создание временной директории
            temp_dir = os.path.join(gettempdir(), "tg_audio_{}".format(user.id))
            os.makedirs(temp_dir, exist_ok=True)
            logger.debug("Временная директория: {}".format(temp_dir))

            # Скачивание файла
            update.message.reply_text('⏳ Обрабатываю аудио...')
            file = context.bot.get_file(audio_file.file_id)
            audio_path = os.path.join(temp_dir, "audio_{}.ogg".format(audio_file.file_id))
            file.download(audio_path)
            logger.info("Аудио сохранено: {} ({} байт)".format(audio_path, os.path.getsize(audio_path)))

            # Валидация файла
            is_valid, message = self.audio_processor.validate_audio_file(audio_path)
            if not is_valid:
                update.message.reply_text('❌ {}'.format(message))
                return

            # Разбиваем аудио на сегменты
            try:
                segment_paths = self.audio_processor.split_audio_into_segments(audio_path)
                logger.info("Создано {} сегментов для обработки".format(len(segment_paths)))
            except Exception as e:
                update.message.reply_text('❌ Ошибка разбиения аудио')
                raise
            
            # Транскрибация всех сегментов
            all_texts = []
            try:
                for i, segment_path in enumerate(segment_paths):
                    logger.info("Обработка сегмента {}/{}".format(i+1, len(segment_paths)))
                    
                    # Обновляем статус для пользователя
                    if len(segment_paths) > 1:
                        update.message.reply_text('⏳ Обрабатываю сегмент {}/{}...'.format(i+1, len(segment_paths)))
                    
                    try:
                        segment_text = self.transcriber.transcribe(segment_path)
                        if segment_text.strip():
                            all_texts.append(segment_text.strip())
                            logger.info("Сегмент {} обработан: {} символов".format(i+1, len(segment_text)))
                        else:
                            logger.warning("Сегмент {} не содержит текста".format(i+1))
                    except Exception as e:
                        logger.error("Ошибка обработки сегмента {}: {}".format(i+1, str(e)))
                        # Продолжаем обработку других сегментов
                        continue
                
                # Объединяем все результаты
                if all_texts:
                    text = ' '.join(all_texts)
                    logger.info("Успешная транскрибация всех сегментов. Общая длина текста: {} символов".format(len(text)))
                else:
                    text = ""
                    logger.warning("Не удалось обработать ни одного сегмента")
                    
            except Exception as e:
                update.message.reply_text('❌ Ошибка распознавания')
                raise

            if not text.strip():
                update.message.reply_text('❌ Не удалось распознать речь')
                return

            # Отправка результата
            self._send_result(update, text, temp_dir)

        except Exception as e:
            logger.error("Ошибка обработки: {}".format(str(e)), exc_info=True)
            
            # Специальная обработка для больших файлов
            if "File is too big" in str(e):
                update.message.reply_text(
                    "❌ Файл слишком большой для обработки. "
                    "Попробуйте отправить файл меньшего размера (до 50 МБ) или "
                    "разбейте длинное аудио на части."
                )
            else:
                update.message.reply_text('⚠️ Внутренняя ошибка сервера')
        
        finally:
            # Очистка временных файлов
            self._cleanup_files(audio_path if 'audio_path' in locals() else None, segment_paths, temp_dir)
    
    def _send_result(self, update, text, temp_dir):
        """Отправляет результат пользователю"""
        if len(text) <= self.config.max_text_length:
            update.message.reply_text("📝 Результат:\n\n{}".format(text))
            logger.info("Текст отправлен сообщением")
        else:
            filename = "Транскрипция_{}.txt".format(datetime.now().strftime('%Y%m%d-%H%M%S'))
            text_path = os.path.join(temp_dir, filename)
            
            try:
                logger.info("Начало записи в {}".format(filename))
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                    logger.info("Записано {} символов".format(len(text)))

                # Отправка файла
                update.message.reply_document(
                    document=open(text_path, 'rb'),
                    filename=filename,
                    caption="📁 Текст слишком длинный"
                )
                logger.info("Файл отправлен")
            
            except Exception as e:
                logger.error("Ошибка записи: {}".format(str(e)), exc_info=True)
                update.message.reply_text('⚠️ Ошибка создания файла')
            
            finally:
                if os.path.exists(text_path):
                    os.remove(text_path)
                    logger.info("Текстовый файл удален")
    
    def _cleanup_files(self, audio_path, segment_paths, temp_dir):
        """Очищает временные файлы"""
        try:
            # Удаляем основной аудиофайл
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Основной аудиофайл удален")
            
            # Удаляем сегменты
            if segment_paths:
                self.audio_processor.cleanup_segments(segment_paths)
            
            # Удаляем временную директорию
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Временная директория удалена: {}".format(temp_dir))
                
        except Exception as e:
            logger.warning("Ошибка очистки файлов: {}".format(str(e)))
    
    def run(self):
        """Запускает бота"""
        try:
            logger.info("🚀 Запуск бота...")
            self.app = Updater(token=self.config.telegram_token, use_context=True)
            self.app.dispatcher.add_handler(CommandHandler("start", self.start_command))
            self.app.dispatcher.add_handler(MessageHandler(Filters.audio | Filters.voice, self.handle_audio))
            self.app.start_polling()
            self.app.idle()
        except Exception as e:
            logger.critical("Критическая ошибка: {}".format(str(e)))
            exit(1)

# Инициализация и запуск бота
try:
    bot = TelegramBot(config)
except Exception as e:
    logger.critical("Невозможно инициализировать бота: {}".format(str(e)))
    exit(1)

def main():
    """Главная функция запуска бота"""
    bot.run()

if __name__ == '__main__':
    main()
