import os
import asyncio
from pathlib import Path
from uuid import uuid4
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = Router()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Инициализация модели VLM
model = None
processor = None
device = None
VLM_SEM = asyncio.Semaphore(1)

# Хранилище временных данных
user_data = {}


class ClothingStates(StatesGroup):
    waiting_for_main_category = State()
    waiting_for_subcategory = State()


# Структура для хранения категорий одежды
class ClothingCategory:
    def __init__(self, id: int, name: str, subcategories=None):
        self.id = id
        self.name = name
        self.subcategories = subcategories or []


# Иерархия категорий одежды
CLOTHING_CATEGORIES = {
    2: ClothingCategory(2, "Одежда", [
        ClothingCategory(3, "Платья", [
            ClothingCategory(4, "Повседневные платья"),
            ClothingCategory(5, "Вечерние платья"),
        ]),
        ClothingCategory(11, "Верхняя одежда", [
            ClothingCategory(21, "Футболки"),
            ClothingCategory(4495, "Свитшоты"),
        ]),
    ]),
    
    35: ClothingCategory(35, "Сумки", [
        ClothingCategory(36, "Тотебы"),
        ClothingCategory(37, "Сумки через плечо"),
    ]),
    
    41: ClothingCategory(41, "Обувь", [
        ClothingCategory(42, "Ботинки"),
        ClothingCategory(49, "Кроссовки"),
    ]),
    
    51: ClothingCategory(51, "Аксессуары"),
    
    60: ClothingCategory(60, "Украшения"),
    
    71: ClothingCategory(71, "Мужская мода"),
}


def run_vlm_sync(image_path: Path, prompt: str) -> str:
    """Синхронный вызов VLM модели"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        do_resize=False,
        return_tensors="pt",
        **(video_kwargs or {}),
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out


async def suggest_vlm(image_path: Path, prompt: str) -> str:
    """Асинхронный вызов VLM модели"""
    async with VLM_SEM:
        return await asyncio.to_thread(run_vlm_sync, image_path, prompt)


def generate_recommendation_with_vlm(image_path: Path, categories: list) -> str:
    """
    Генерация рекомендаций с использованием VLM модели на основе выбранных категорий
    """
    try:
        # Формируем промпт на основе выбранных категорий
        categories_text = ", ".join([cat['name'] for cat in categories])
        prompt = f"Опиши подробно какие предметы из категорий: {categories_text} можно добавить к этому образу, чтобы дополнить его. Учти текущий стиль на фото."
        
        # Вызываем VLM модель
        result = run_vlm_sync(image_path, prompt)
        
        if not result or len(result.strip()) == 0:
            return "⚠️ Модель не смогла сгенерировать рекомендации. Пожалуйста, попробуйте с другим изображением."
        
        # Форматируем результат
        formatted_recommendation = f"🎯 **Рекомендации по выбранным категориям:**\n\n"
        formatted_recommendation += f"📋 *Категории:* {categories_text}\n\n"
        formatted_recommendation += result
        formatted_recommendation += "\n\n---\n*Рекомендация сгенерирована моделью Qwen VL*"
        
        return formatted_recommendation
        
    except Exception as e:
        logger.error(f"Ошибка в generate_recommendation_with_vlm: {e}")
        return f"❌ Произошла ошибка при анализе: {str(e)}. Пожалуйста, попробуйте снова."


async def generate_recommendation_async(image_path: Path, categories: list) -> str:
    """Асинхронная версия генерации рекомендаций"""
    try:
        categories_text = ", ".join([cat['name'] for cat in categories])
        prompt = f"Опиши подробно какие предметы из категорий: {categories_text} можно добавить к этому образу, чтобы дополнить его. Учти текущий стиль на фото."
        
        result = await suggest_vlm(image_path, prompt)
        
        if not result or len(result.strip()) == 0:
            return "⚠️ Модель не смогла сгенерировать рекомендации. Пожалуйста, попробуйте с другим изображением."
        
        formatted_recommendation = f"🎯 **Рекомендации по выбранным категориям:**\n\n"
        formatted_recommendation += f"📋 *Категории:* {categories_text}\n\n"
        formatted_recommendation += result
        formatted_recommendation += "\n\n---\n*Рекомендация сгенерирована моделью Qwen VL*"
        
        return formatted_recommendation
        
    except Exception as e:
        logger.error(f"Ошибка в generate_recommendation_async: {e}")
        return f"❌ Произошла ошибка при анализе: {str(e)[:100]}... Пожалуйста, попробуйте снова."


def get_main_categories_keyboard():
    """Клавиатура с основными категориями"""
    builder = InlineKeyboardBuilder()
    
    main_categories = [
        (2, "👕 Одежда"),
        (35, "👜 Сумки"),
        (41, "👟 Обувь"),
        (51, "🧣 Аксессуары"),
        (60, "💍 Украшения"),
        (71, "👔 Мужская мода"),
    ]
    
    for cat_id, name in main_categories:
        builder.button(text=name, callback_data=f"cat_{cat_id}")
    
    builder.adjust(2, 2, 2)
    return builder.as_markup()


def get_subcategories_keyboard(category_id: int):
    """Клавиатура с подкатегориями для выбранной категории"""
    builder = InlineKeyboardBuilder()
    
    if category_id in CLOTHING_CATEGORIES:
        category = CLOTHING_CATEGORIES[category_id]
        
        if category.subcategories:
            for subcat in category.subcategories:
                builder.button(text=f"📁 {subcat.name}", callback_data=f"cat_{subcat.id}")
            
            builder.button(text=f"✅ Выбрать '{category.name}'", callback_data=f"select_{category_id}")
        else:
            builder.button(text=f"✅ Выбрать '{category.name}'", callback_data=f"select_{category_id}")
    
    builder.button(text="⬅️ Назад к категориям", callback_data="back_to_main")
    builder.adjust(1)
    
    return builder.as_markup()


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Пришли фото аутфита — в ответ пришлю, что можно добавить (фото + текст)."
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Ок, сбросил. Напиши /start, чтобы начать заново.")


@router.message(F.photo)
async def on_photo(message: Message, state: FSMContext) -> None:
    photo = message.photo[-1]
    
    # Сохраняем фото
    in_path = UPLOAD_DIR / f"{uuid4().hex}.jpg"
    await message.bot.download(photo, destination=in_path)
    
    # Сохраняем информацию о фото пользователя
    user_data[message.from_user.id] = {
        'photo_path': str(in_path),
        'photo_id': photo.file_id,
        'current_category': None,
        'selected_items': []
    }
    
    await message.answer(
        "🎨 Выберите категорию одежды для получения рекомендаций:",
        reply_markup=get_main_categories_keyboard()
    )
    
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(F.data.startswith("cat_"), ClothingStates.waiting_for_main_category)
async def process_category_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    
    if user_id not in user_data:
        await callback.message.answer("Сессия устарела. Пожалуйста, отправьте фото заново.")
        await state.clear()
        return
    
    category_id = int(callback.data.split("_")[1])
    user_data[user_id]['current_category'] = category_id
    
    if category_id in CLOTHING_CATEGORIES:
        category = CLOTHING_CATEGORIES[category_id]
        
        if category.subcategories:
            await callback.message.edit_text(
                f"Вы выбрали: {category.name}\nВыберите подкатегорию:",
                reply_markup=get_subcategories_keyboard(category_id)
            )
            await state.set_state(ClothingStates.waiting_for_subcategory)
        else:
            await process_selection(callback, category_id, category.name)


@router.callback_query(F.data.startswith("cat_"), ClothingStates.waiting_for_subcategory)
async def process_subcategory_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    
    if user_id not in user_data:
        await callback.message.answer("Сессия устарела. Пожалуйста, отправьте фото заново.")
        await state.clear()
        return
    
    subcategory_id = int(callback.data.split("_")[1])
    
    # Находим имя подкатегории
    subcategory_name = "Неизвестная категория"
    parent_id = user_data[user_id].get('current_category')
    
    if parent_id in CLOTHING_CATEGORIES:
        parent = CLOTHING_CATEGORIES[parent_id]
        for subcat in parent.subcategories:
            if subcat.id == subcategory_id:
                subcategory_name = subcat.name
                break
    
    if subcategory_id in CLOTHING_CATEGORIES:
        category = CLOTHING_CATEGORIES[subcategory_id]
        
        if category.subcategories:
            user_data[user_id]['current_category'] = subcategory_id
            await callback.message.edit_text(
                f"Вы выбрали: {subcategory_name}\nВыберите подкатегорию:",
                reply_markup=get_subcategories_keyboard(subcategory_id)
            )
        else:
            await process_selection(callback, subcategory_id, subcategory_name)
    else:
        await process_selection(callback, subcategory_id, subcategory_name)


@router.callback_query(F.data.startswith("select_"))
async def process_selection(callback: CallbackQuery, category_id: int = None, category_name: str = None):
    user_id = callback.from_user.id
    
    if user_id not in user_data:
        await callback.message.answer("Сессия устарела. Пожалуйста, отправьте фото заново.")
        return
    
    if callback and not category_id:
        category_id = int(callback.data.split("_")[1])
        if category_id in CLOTHING_CATEGORIES:
            category_name = CLOTHING_CATEGORIES[category_id].name
    
    if 'selected_items' not in user_data[user_id]:
        user_data[user_id]['selected_items'] = []
    
    user_data[user_id]['selected_items'].append({
        'id': category_id,
        'name': category_name
    })
    
    await callback.message.edit_text(f"✅ Выбрано: {category_name}")
    await show_selection_options(callback)


async def show_selection_options(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    builder = InlineKeyboardBuilder()
    builder.button(text="➕ Добавить еще элемент", callback_data="add_more")
    
    selected_count = len(user_data[user_id].get('selected_items', []))
    builder.button(text=f"🔍 Анализировать ({selected_count})", callback_data="analyze_selected")
    
    builder.button(text="🗑️ Очистить выбор", callback_data="clear_selection")
    builder.adjust(1)
    
    selected_items = user_data[user_id].get('selected_items', [])
    if selected_items:
        items_list = "\n".join([f"• {item['name']}" for item in selected_items])
        message_text = f"📋 Вы выбрали:\n{items_list}\n\nЧто хотите сделать дальше?"
    else:
        message_text = "📋 Вы еще ничего не выбрали.\n\nЧто хотите сделать дальше?"
    
    await callback.message.answer(
        message_text,
        reply_markup=builder.as_markup()
    )


@router.callback_query(F.data == "add_more")
async def add_more_items(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "🎨 Выберите еще одну категорию одежды:",
        reply_markup=get_main_categories_keyboard()
    )
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(F.data == "analyze_selected")
async def analyze_selected_items(callback: CallbackQuery):
    user_id = callback.from_user.id
    
    if user_id not in user_data:
        await callback.message.answer("Сессия устарела. Пожалуйста, отправьте фото заново.")
        return
    
    selected_items = user_data[user_id].get('selected_items', [])
    
    if not selected_items:
        await callback.message.answer("Вы ничего не выбрали. Пожалуйста, выберите хотя бы одну категорию.")
        return
    
    await callback.message.edit_text("🔍 Начинаю анализ выбранных категорий с использованием VLM модели...")
    
    photo_path = Path(user_data[user_id]['photo_path'])
    photo_id = user_data[user_id]['photo_id']
    
    # Генерируем рекомендации с использованием VLM модели
    caption = await generate_recommendation_async(photo_path, selected_items)
    
    try:
        await callback.message.answer_photo(
            photo=photo_id,
            caption=caption,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Ошибка при отправке фото: {e}")
        # Если не удалось отправить фото, отправляем только текст
        await callback.message.answer(
            caption,
            parse_mode=ParseMode.MARKDOWN
        )
    
    await callback.message.answer(
        "🔄 Хотите проанализировать другие категории одежды?",
        reply_markup=get_main_categories_keyboard()
    )
    
    # Очищаем выбранные элементы
    user_data[user_id]['selected_items'] = []


@router.callback_query(F.data == "clear_selection")
async def clear_selection(callback: CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    
    if user_id in user_data:
        user_data[user_id]['selected_items'] = []
    
    await callback.message.edit_text("✅ Выбор очищен. Давайте начнем заново:")
    await callback.message.answer(
        "🎨 Выберите категорию одежды для получения рекомендаций:",
        reply_markup=get_main_categories_keyboard()
    )
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(F.data == "back_to_main")
async def back_to_main_categories(callback: CallbackQuery, state: FSMContext):
    await callback.message.edit_text(
        "🎨 Выберите категорию одежды для получения рекомендаций:",
        reply_markup=get_main_categories_keyboard()
    )
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.message(ClothingStates.waiting_for_main_category)
@router.message(ClothingStates.waiting_for_subcategory)
async def handle_wrong_input(message: Message):
    await message.answer("Пожалуйста, используйте кнопки для выбора категории одежды.")


async def main() -> None:
    global model, processor, device
    
    # Инициализация VLM модели
    try:
        logger.info("Инициализация VLM модели...")
        ckpt = "Qwen/Qwen3-VL-4B-Thinking"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство: {device}")
        
        model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        
        processor = AutoProcessor.from_pretrained(ckpt)
        logger.info("VLM модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке VLM модели: {e}")
        logger.warning("Бот будет работать без VLM модели")
        model = None
        processor = None
    
    # Инициализация бота
    token = os.getenv("BOT_TOKEN", "") #токен бота
    if not token:
        raise RuntimeError("Set BOT_TOKEN env var or add it to the code")
    
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    
    logger.info("Бот запущен и готов к работе")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())