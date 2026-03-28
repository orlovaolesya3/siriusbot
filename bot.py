import os
import re
import json
import asyncio
import logging
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

import sglang as sgl

from PIL import Image, ImageOps

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.filters.callback_data import CallbackData
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("stylist_bot")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FASHION_DATA_PATH = os.getenv("FASHION_DATA_PATH", "/home/borodinro/fashion_attribution/data")
FASHION_DEVICE = os.getenv("FASHION_DEVICE", "cpu")
DEBUG_JSON = os.getenv("DEBUG_JSON", "0") == "1"

VLM_SEM = asyncio.Semaphore(1)
SEARCH_SEM = asyncio.Semaphore(2)
USER_LOCKS: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

router = Router()
SEARCH_ENGINE = None


async def send_info(chat_id: int, bot: Bot, text: str, parse_mode: ParseMode = ParseMode.HTML) -> None:
    logger.info("Send to %s: %s", chat_id, text)
    await bot.send_message(chat_id, text, parse_mode=parse_mode)


async def send_error(chat_id: int, bot: Bot, text: str, parse_mode: ParseMode = ParseMode.HTML) -> None:
    logger.error("Send to %s: %s", chat_id, text)
    await bot.send_message(chat_id, text, parse_mode=parse_mode)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


async def safe_delete_message(bot: Bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass


async def delete_previous_bot_messages(chat_id: int, bot: Bot, keep_ids: set[int]) -> None:
    try:
        async for message in bot.get_chat_history(chat_id):
            if message.from_user.id == bot.id and message.message_id not in keep_ids:
                await safe_delete_message(bot, chat_id, message.message_id)
                await asyncio.sleep(0.1)
    except Exception:
        pass


def resize_image_if_needed(image_path: Path, max_size: int = 1024, jpeg_quality: int = 90) -> None:
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))

            suffix = image_path.suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                img.save(image_path, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            else:
                img.save(image_path)
    except Exception:
        logger.exception("Resize failed for %s", image_path)


def clean_thinking_tags(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


def extract_first_json_object(text: str) -> str | None:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start: i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict) and "items" in obj:
                        return json.dumps(obj, ensure_ascii=False)
                except Exception:
                    return None
    return None


def session_item_key(item: dict[str, Any]) -> str:
    name = str(item.get("name", "")).strip().lower()
    iid = str(item.get("id", "")).strip()
    return f"{iid}:{name}"


class FashionSearchEngine:
    def __init__(self, base_path=FASHION_DATA_PATH, device=FASHION_DEVICE):
        self.base_path = Path(base_path)
        self.index_dir = self.base_path / "unified_index"
        self.images_dir = self.base_path / "good_images"
        self.device = device

        logger.info("[SearchEngine] init on %s", self.device)
        self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)

        logger.info("[SearchEngine] loading vectors/metadata...")
        self.vectors = torch.from_numpy(np.load(self.index_dir / "all_vectors.npy")).to(self.device)
        with open(self.index_dir / "all_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        null_prompt = (
            "Category: ; Type: ; Formality: ; Gender: ; Color: ; Material: ; Silhouette: ; "
            "Season: ; Style: ; Print: ; Fit: ; Length: ; Sleeve: "
        )
        self.null_embedding = self.model.encode(null_prompt, convert_to_tensor=True, device=self.device)

        self.category_index: dict[str, list[int]] = {}
        self.type_index: dict[str, list[int]] = {}

        for idx, meta in enumerate(self.metadata):
            desc = meta.get("desc", "")
            if not isinstance(desc, str):
                continue
            fields: dict[str, str] = {}
            for part in desc.split(";"):
                if ":" in part:
                    key, value = part.split(":", 1)
                    fields[key.strip().lower()] = value.strip().lower()
            category = fields.get("category")
            typ = fields.get("type")
            if category:
                self.category_index.setdefault(category, []).append(idx)
            if typ:
                self.type_index.setdefault(typ, []).append(idx)

    def parse_json_to_desc(self, item_dict: dict) -> str:
        mapping = [
            ("Category", "category_name"),
            ("Type", "type"),
            ("Formality", "formality"),
            ("Gender", "gender"),
            ("Color", "color"),
            ("Material", "material"),
            ("Silhouette", "silhouette"),
            ("Season", "season"),
            ("Style", "style"),
            ("Print", "print"),
            ("Fit", "fit"),
            ("Length", "length"),
            ("Sleeve", "sleeve"),
        ]
        parts: list[str] = []
        for label, key in mapping:
            value = item_dict.get(key, "N/A")
            parts.append(f"{label}: {value}")
        return "; ".join(parts)

    def get_image_path(self, set_id: str, index: str) -> str | None:
        folder = self.images_dir / set_id
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            path = folder / f"{index}{ext}"
            if path.exists():
                return str(path)
        return None

    def find_best_match(self, query_json_str: str):
        data = json.loads(query_json_str)
        items = data.get("items", [])
        if isinstance(items, dict):
            items = [items]

        results: list[dict[str, Any]] = []
        for item in items:
            desc_query = self.parse_json_to_desc(item)
            query_emb = self.model.encode(desc_query, convert_to_tensor=True, device=self.device)
            clean_emb = query_emb - self.null_embedding

            category = (item.get("category_name") or "").strip().lower()
            typ = (item.get("type") or "").strip().lower()

            subset: set[int] | None = None
            if category and category != "n/a" and category in self.category_index:
                subset = set(self.category_index[category])
            if typ and typ != "n/a" and typ in self.type_index:
                tset = set(self.type_index[typ])
                subset = tset if subset is None else subset.intersection(tset)

            if subset is not None and not subset:
                results.append(
                    {
                        "query_category": item.get("category_name", "N/A"),
                        "match_score": "N/A",
                        "set_id": "",
                        "index": "",
                        "photo": None,
                        "original_desc": "No items found for this category/type",
                    }
                )
                continue

            subset_list = sorted(subset) if subset is not None else None
            vecs = self.vectors if subset_list is None else self.vectors[subset_list]

            similarities = util.cos_sim(clean_emb, vecs)[0]
            best_local = torch.argmax(similarities).item()
            best_score = float(similarities[best_local])

            best_idx = subset_list[best_local] if subset_list is not None else best_local
            match_data = self.metadata[best_idx]
            sid = str(match_data.get("set_id"))
            idx = str(match_data.get("index"))
            photo_path = self.get_image_path(sid, idx)

            results.append(
                {
                    "query_category": item.get("category_name", "N/A"),
                    "match_score": f"{best_score * 100:.2f}%",
                    "set_id": sid,
                    "index": idx,
                    "photo": photo_path,
                    "original_desc": match_data.get("desc", ""),
                }
            )
        return results


async def search_matches_async(query_json_str: str):
    async with SEARCH_SEM:
        if SEARCH_ENGINE is None:
            raise RuntimeError("SEARCH_ENGINE is not initialized")
        return await asyncio.to_thread(SEARCH_ENGINE.find_best_match, query_json_str)


@sgl.function
def clothing_analyst(s, image_path: str, prompt_text: str, temperature: float = 0.4, max_tokens: int = 8192):
    system_prompt = """You are a professional stylist and buyer.
You will recommend EXACTLY ONE item per requested category based on the outfit in the image.

OUTPUT RULES (STRICT):
- Output must be a VALID JSON OBJECT in STRICTLY ONE LINE.
- Do not output any extra text before or after the JSON.
- Root object must be: {"items":[...]}
- "items" length MUST equal the number of requested categories.
- Order of "items[i]" MUST match the order of requested categories in the user message.
- Each "items[i]" must be an object with EXACTLY these keys and no others:
  category_name, type, formality, gender, color, material, silhouette, season, style, print, fit, length, sleeve
- Every value must be a string.
- IMPORTANT: category_name must EXACTLY match the provided category_name strings (do not translate them).
- For all other fields, use English words.
- Use "N/A" only if necessary and avoid using it for more than half the fields.
- No markdown, no comments, no trailing commas, no line breaks.
"""
    s += sgl.system(system_prompt)
    s += sgl.user(sgl.image(image_path) + "\n\n" + prompt_text)
    s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, temperature=temperature))


def run_vlm_sglang_sync(image_path: Path, prompt: str, temperature: float = 0.4, max_tokens: int = 8192) -> str:
    try:
        resize_image_if_needed(image_path, max_size=1024)

        state = clothing_analyst.run(
            image_path=str(image_path),
            prompt_text=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = state["response"]
        cleaned = clean_thinking_tags(raw)
        json_line = extract_first_json_object(cleaned)
        if not json_line:
            logger.warning("VLM returned non-JSON (first 300): %r", cleaned[:300])
            return ""
        return json_line
    except Exception:
        logger.exception("SGLang Runtime Error")
        return ""


async def suggest_vlm(image_path: Path, prompt: str, temperature: float = 0.4, max_tokens: int = 8192) -> str:
    async with VLM_SEM:
        return await asyncio.to_thread(run_vlm_sglang_sync, image_path, prompt, temperature, max_tokens)


async def generate_recommendation_json(image_path: Path, categories: list[dict[str, Any]]) -> str:
    categories_payload = [{"category_name": c["name"]} for c in categories]
    prompt = (
            "Requested categories (strict order). Do not translate category_name:\n"
            + json.dumps(categories_payload, ensure_ascii=False)
            + "\nReturn JSON exactly as required by SYSTEM."
    )
    for temp in (0.4, 0.2):
        js = await suggest_vlm(image_path, prompt, temperature=temp, max_tokens=8192)
        if js:
            return js
    return ""


class ClothingStates(StatesGroup):
    waiting_for_photo = State()
    waiting_for_main_category = State()
    waiting_for_subcategory = State()
    waiting_for_item_selection = State()
    waiting_for_custom_category = State()


@dataclass
class ClothingCategory:
    id: int
    name: str
    subcategories: list["ClothingCategory"] = field(default_factory=list)


CLOTHING_CATEGORIES: dict[int, ClothingCategory] = {
    2: ClothingCategory(2, "Верх", [
        ClothingCategory(200, "Топы", [
            ClothingCategory(21, "Футболки"),
            ClothingCategory(201, "Лонгсливы"),
            ClothingCategory(24, "Топы"),
        ]),
        ClothingCategory(202, "Рубашки и блузки", [
            ClothingCategory(22, "Рубашки"),
            ClothingCategory(23, "Блузки"),
        ]),
        ClothingCategory(203, "Трикотаж", [
            ClothingCategory(4495, "Худи и свитшоты"),
            ClothingCategory(204, "Свитеры"),
            ClothingCategory(27, "Кардиганы"),
        ]),
        ClothingCategory(11, "Верхняя одежда", [
            ClothingCategory(205, "Куртки"),
            ClothingCategory(206, "Пальто"),
            ClothingCategory(207, "Пуховики"),
        ]),
        ClothingCategory(3, "Платья и комбинезоны", [
            ClothingCategory(4, "Платья"),
            ClothingCategory(83, "Комбинезоны"),
        ]),
        ClothingCategory(216, "Жакеты и жилеты", [
            ClothingCategory(25, "Пиджаки и жакеты"),
            ClothingCategory(26, "Жилеты"),
        ]),
    ]),

    100: ClothingCategory(100, "Низ", [
        ClothingCategory(102, "Джинсы"),
        ClothingCategory(208, "Брюки"),
        ClothingCategory(209, "Юбки"),
        ClothingCategory(210, "Шорты"),
    ]),

    41: ClothingCategory(41, "Обувь", [
        ClothingCategory(49, "Кроссовки"),
        ClothingCategory(42, "Ботинки"),
        ClothingCategory(52, "Сапоги"),
        ClothingCategory(50, "Туфли"),
        ClothingCategory(53, "Лоферы"),
        ClothingCategory(51, "Сандалии"),
        ClothingCategory(54, "Босоножки"),
    ]),

    35: ClothingCategory(35, "Сумки", [
        ClothingCategory(37, "Сумки через плечо"),
        ClothingCategory(215, "Сумки в руку"),
        ClothingCategory(39, "Рюкзаки"),
        ClothingCategory(40, "Поясные сумки"),
    ]),

    60: ClothingCategory(60, "Аксессуары", [
        ClothingCategory(61, "Головные уборы"),
        ClothingCategory(62, "Шарфы"),
        ClothingCategory(63, "Ремни"),
        ClothingCategory(65, "Очки"),
        ClothingCategory(64, "Перчатки"),
        ClothingCategory(66, "Зонты"),
    ]),

    70: ClothingCategory(70, "Украшения", [
        ClothingCategory(71, "Серьги"),
        ClothingCategory(72, "Колье и подвески"),
        ClothingCategory(73, "Браслеты"),
        ClothingCategory(74, "Кольца"),
        ClothingCategory(75, "Часы"),
    ]),

    80: ClothingCategory(80, "Костюмы", [
        ClothingCategory(82, "Костюмы-двойки"),
    ]),
}


def build_category_maps() -> tuple[dict[int, ClothingCategory], dict[int, int | None]]:
    by_id: dict[int, ClothingCategory] = {}
    parent: dict[int, int | None] = {}

    def walk(node: ClothingCategory, parent_id: int | None):
        by_id[node.id] = node
        parent[node.id] = parent_id
        for ch in node.subcategories:
            walk(ch, node.id)

    for root in CLOTHING_CATEGORIES.values():
        walk(root, None)
    return by_id, parent


CAT_BY_ID, PARENT_BY_ID = build_category_maps()


class MainCatCb(CallbackData, prefix="mc"):
    id: int


class SubCatCb(CallbackData, prefix="sc"):
    id: int


class ItemCb(CallbackData, prefix="it"):
    id: int


class SelectCatCb(CallbackData, prefix="cat"):
    id: int


class ActionCb(CallbackData, prefix="act"):
    action: str
    arg: int = 0


def get_main_categories_keyboard():
    builder = InlineKeyboardBuilder()
    main_categories = [
        (2, "👕 Верх"),
        (100, "👖 Низ"),
        (80, "👔 Костюмы"),
        (35, "👜 Сумки"),
        (41, "👟 Обувь"),
        (60, "🧣 Аксессуары"),
        (70, "💍 Украшения"),
    ]
    for cat_id, name in main_categories:
        builder.button(text=name, callback_data=MainCatCb(id=cat_id))
    builder.button(text="➕ Добавить своё", callback_data=ActionCb(action="custom"))
    builder.adjust(2, 2, 2, 1)
    return builder.as_markup()


def get_subcategories_keyboard(category_id: int):
    builder = InlineKeyboardBuilder()
    category = CLOTHING_CATEGORIES.get(category_id)
    if category:
        for subcat in category.subcategories:
            builder.button(text=f"📂 {subcat.name}", callback_data=SubCatCb(id=subcat.id))
        builder.button(text=f"✅ Выбрать всю категорию '{category.name}'", callback_data=SelectCatCb(id=category.id))
    builder.button(text="⬅️ Назад", callback_data=ActionCb(action="back_main"))
    builder.adjust(1)
    return builder.as_markup()


def get_items_keyboard(parent_id: int, subcategory_id: int):
    builder = InlineKeyboardBuilder()
    subcat = CAT_BY_ID.get(subcategory_id)
    if subcat and subcat.subcategories:
        for item in subcat.subcategories:
            builder.button(text=f"• {item.name}", callback_data=ItemCb(id=item.id))
        builder.button(text=f"✅ Выбрать все '{subcat.name}'", callback_data=SelectCatCb(id=subcat.id))

    builder.button(text="⬅️ Назад", callback_data=ActionCb(action="back_sub", arg=parent_id))
    builder.adjust(1)
    return builder.as_markup()


def get_selection_options_keyboard():
    builder = InlineKeyboardBuilder()
    builder.button(text="➕ Добавить еще", callback_data=ActionCb(action="add_more"))
    builder.button(text="🔍 Анализировать", callback_data=ActionCb(action="analyze"))
    builder.button(text="🗑️ Очистить выбор", callback_data=ActionCb(action="clear"))
    builder.adjust(1)
    return builder.as_markup()


async def get_session(state: FSMContext) -> dict[str, Any]:
    data = await state.get_data()
    return data


async def set_session(state: FSMContext, **kwargs) -> None:
    await state.update_data(**kwargs)


async def ensure_session_has_photo(message_or_cb: Message | CallbackQuery, state: FSMContext) -> bool:
    data = await state.get_data()
    if not data.get("photo_path"):
        if isinstance(message_or_cb, CallbackQuery):
            await message_or_cb.answer("Сначала пришли фото 📷", show_alert=True)
        else:
            await send_error(message_or_cb.chat.id, message_or_cb.bot, "Сначала отправь фото 📷")
        return False
    return True


async def update_selection_message(chat_id: int, bot: Bot, state: FSMContext) -> None:
    data = await state.get_data()
    menu_message_id = data.get("menu_message_id")
    items: list[dict[str, Any]] = data.get("selected_items", [])
    
    keep_ids = {menu_message_id} if menu_message_id else set()
    await delete_previous_bot_messages(chat_id, bot, keep_ids)

    items_list = "\n".join([f"• {i['name']}" for i in items]) or "—"
    text = (
        "Ваш выбор категорий:\n\n"
        f"📋 <b>Выбрано:</b>\n{items_list}\n\n"
        "Добавляйте еще, анализируйте или очищайте список."
    )

    if menu_message_id:
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=menu_message_id,
                text=text,
                reply_markup=get_selection_options_keyboard(),
                parse_mode=ParseMode.HTML,
            )
            return
        except Exception:
            pass

    msg = await bot.send_message(chat_id, text, reply_markup=get_selection_options_keyboard(),
                                 parse_mode=ParseMode.HTML)
    await set_session(state, menu_message_id=msg.message_id)


async def add_selected_item(state: FSMContext, item_id: int | str, name: str, is_custom: bool = False) -> bool:
    data = await state.get_data()
    items: list[dict[str, Any]] = data.get("selected_items", [])
    new_item = {"id": item_id, "name": name, "is_custom": is_custom}

    existing_keys = {session_item_key(i) for i in items}
    if session_item_key(new_item) in existing_keys:
        return False

    items.append(new_item)
    await set_session(state, selected_items=items)
    return True


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(ClothingStates.waiting_for_photo)
    await send_info(
        message.chat.id,
        message.bot,
        "👋 <b>Привет! Я твой персональный стилист!</b>\n\n"
        "📸 <b>Отправь мне фото аутфита</b>, а я проанализирую твои элементы одежды и найду подходящие варианты в базе\n"
        "<i>Просто отправь фото и следуй инструкциям 👇</i>",
    )


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if data.get("photo_path"):
        safe_unlink(Path(data["photo_path"]))
    
    await state.clear()
    await state.set_state(ClothingStates.waiting_for_photo)
    
    await send_info(
        message.chat.id,
        message.bot,
        "🔄 <b>Сессия сброшена!</b>\n\n"
        "📸 <b>Готов к новому фото!</b>\n"
        "Отправь мне изображение аутфита для анализа 👇",
    )


@router.message(F.photo)
async def on_photo(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    if data.get("photo_path"):
        safe_unlink(Path(data["photo_path"]))
    
    photo = message.photo[-1]
    in_path = UPLOAD_DIR / f"{uuid4().hex}.jpg"
    await message.bot.download(photo, destination=in_path)

    await delete_previous_bot_messages(message.chat.id, message.bot, {message.message_id})

    confirmation_msg = await message.answer(
        "✅ <b>Фото получено!</b>\n\n"
        "📋 <b>Теперь выбери категории одежды для поиска:</b>\n"
        "Нажми на кнопку с нужной категорией ниже 👇",
        reply_markup=get_main_categories_keyboard(),
        parse_mode=ParseMode.HTML
    )

    await set_session(
        state,
        photo_path=str(in_path),
        photo_id=photo.file_id,
        selected_items=[],
        menu_message_id=confirmation_msg.message_id,
        custom_prompt_message_id=None,
    )
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.message(ClothingStates.waiting_for_photo)
async def handle_text_without_photo(message: Message, state: FSMContext) -> None:
    if message.text and not message.text.startswith('/'):
        await send_info(
            message.chat.id,
            message.bot,
            "📸 <b>Отправь мне фото аутфита!</b>\n\n"
            "Я проанализирую одежду на фото и найду похожие товары.\n"
            "Просто отправь изображение 👇",
            parse_mode=ParseMode.HTML
        )


@router.callback_query(ActionCb.filter(F.action == "custom"))
async def start_custom_category_input(callback: CallbackQuery, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    data = await state.get_data()
    menu_mid = data.get("menu_message_id")
    if menu_mid:
        try:
            await callback.bot.edit_message_reply_markup(callback.message.chat.id, menu_mid, reply_markup=None)
        except Exception:
            pass

    prompt_msg = await callback.message.answer("✏️ Напиши свою категорию, например: <b>очки авиаторы</b>",
                                               parse_mode=ParseMode.HTML)
    await set_session(state, custom_prompt_message_id=prompt_msg.message_id)
    await state.set_state(ClothingStates.waiting_for_custom_category)
    await callback.answer()


@router.message(ClothingStates.waiting_for_custom_category)
async def process_custom_category_input(message: Message, state: FSMContext):
    if not await ensure_session_has_photo(message, state):
        return

    text = (message.text or "").strip()
    if not text or len(text) > 100:
        await send_error(message.chat.id, message.bot, "Слишком длинно или пусто. Давай покороче и по делу 🙂")
        return

    data = await state.get_data()
    prompt_mid = data.get("custom_prompt_message_id")
    if prompt_mid:
        try:
            await message.bot.delete_message(message.chat.id, prompt_mid)
        except Exception:
            pass
    try:
        await message.delete()
    except Exception:
        pass

    custom_id = f"custom:{uuid4().hex}"
    ok = await add_selected_item(state, custom_id, text, is_custom=True)
    if not ok:
        await send_info(message.chat.id, message.bot, "Такое уже добавлено в список 👀")
    await update_selection_message(message.chat.id, message.bot, state)
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(MainCatCb.filter(), ClothingStates.waiting_for_main_category)
async def process_main_category_selection(callback: CallbackQuery, callback_data: MainCatCb, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    cat_id = callback_data.id
    category = CLOTHING_CATEGORIES.get(cat_id)
    if not category:
        await callback.answer("Не нашёл такую категорию 🤔", show_alert=True)
        return

    try:
        await callback.message.edit_text(
            f"📂 <b>{category.name}</b>\nВыбери подкатегорию:",
            reply_markup=get_subcategories_keyboard(cat_id),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        await callback.message.answer(
            f"📂 <b>{category.name}</b>\nВыбери подкатегорию:",
            reply_markup=get_subcategories_keyboard(cat_id),
            parse_mode=ParseMode.HTML,
        )
    await state.set_state(ClothingStates.waiting_for_subcategory)
    await callback.answer()


@router.callback_query(SubCatCb.filter(), ClothingStates.waiting_for_subcategory)
async def process_subcategory_selection(callback: CallbackQuery, callback_data: SubCatCb, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    sub_id = callback_data.id
    subcat = CAT_BY_ID.get(sub_id)
    if not subcat:
        await callback.answer("Не нашёл подкатегорию 🤔", show_alert=True)
        return
    if subcat.subcategories:
        parent_id = PARENT_BY_ID.get(sub_id)
        if parent_id is None:
            parent_id = sub_id
        try:
            await callback.message.edit_text(
                f"📁 <b>{subcat.name}</b>\nВыбери конкретный тип:",
                reply_markup=get_items_keyboard(parent_id, sub_id),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            await callback.message.answer(
                f"📁 <b>{subcat.name}</b>\nВыбери конкретный тип:",
                reply_markup=get_items_keyboard(parent_id, sub_id),
                parse_mode=ParseMode.HTML,
            )
        await state.set_state(ClothingStates.waiting_for_item_selection)
        await callback.answer()
        return

    ok = await add_selected_item(state, sub_id, subcat.name)
    await callback.answer("✅ Добавлено" if ok else "⚠️ Уже в списке")
    await update_selection_message(callback.message.chat.id, callback.message.bot, state)
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(ItemCb.filter(), ClothingStates.waiting_for_item_selection)
async def process_item_selection(callback: CallbackQuery, callback_data: ItemCb, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    item_id = callback_data.id
    item = CAT_BY_ID.get(item_id)
    if not item:
        await callback.answer("Не нашёл пункт 🤔", show_alert=True)
        return

    ok = await add_selected_item(state, item_id, item.name)
    await callback.answer(f"✅ Добавлено: {item.name}" if ok else "⚠️ Уже в списке")
    await update_selection_message(callback.message.chat.id, callback.message.bot, state)
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(SelectCatCb.filter())
async def process_select_cat(callback: CallbackQuery, callback_data: SelectCatCb, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    cat_id = callback_data.id
    cat = CAT_BY_ID.get(cat_id)
    if not cat:
        await callback.answer("Не нашёл категорию 🤔", show_alert=True)
        return

    ok = await add_selected_item(state, cat_id, cat.name)
    await callback.answer(f"✅ Добавлено: {cat.name}" if ok else "⚠️ Уже в списке")
    await update_selection_message(callback.message.chat.id, callback.message.bot, state)
    await state.set_state(ClothingStates.waiting_for_main_category)


@router.callback_query(ActionCb.filter(F.action == "back_main"))
async def back_to_main(callback: CallbackQuery, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    await callback.message.edit_text("Выбери категорию:", reply_markup=get_main_categories_keyboard())
    await state.set_state(ClothingStates.waiting_for_main_category)
    await callback.answer()


@router.callback_query(ActionCb.filter(F.action == "back_sub"))
async def back_to_sub(callback: CallbackQuery, callback_data: ActionCb, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    parent_id = callback_data.arg
    parent = CAT_BY_ID.get(parent_id)
    if not parent:
        await callback.answer("Назад не получилось 🤷", show_alert=True)
        return
    root_id = parent_id if parent_id in CLOTHING_CATEGORIES else (PARENT_BY_ID.get(parent_id) or parent_id)
    if root_id not in CLOTHING_CATEGORIES:
        root_id = parent_id if parent_id in CLOTHING_CATEGORIES else parent_id

    await callback.message.edit_text(
        f"📂 <b>{CLOTHING_CATEGORIES.get(root_id, parent).name}</b>\nВыбери подкатегорию:",
        reply_markup=get_subcategories_keyboard(root_id),
        parse_mode=ParseMode.HTML,
    )
    await state.set_state(ClothingStates.waiting_for_subcategory)
    await callback.answer()


@router.callback_query(ActionCb.filter(F.action == "add_more"))
async def add_more(callback: CallbackQuery, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    await callback.message.edit_text("Ок, добавим ещё. Выбери категорию:", reply_markup=get_main_categories_keyboard())
    await state.set_state(ClothingStates.waiting_for_main_category)
    await callback.answer()


@router.callback_query(ActionCb.filter(F.action == "clear"))
async def clear_selection(callback: CallbackQuery, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    await set_session(state, selected_items=[])
    await callback.message.edit_text(
        "Список очищен 🧹\n\nВыбери категорию заново:",
        reply_markup=get_main_categories_keyboard(),
    )
    await state.set_state(ClothingStates.waiting_for_main_category)
    await callback.answer()


@router.callback_query(ActionCb.filter(F.action == "analyze"))
async def analyze_selected_items(callback: CallbackQuery, state: FSMContext):
    if not await ensure_session_has_photo(callback, state):
        return
    user_id = callback.from_user.id
    async with USER_LOCKS[user_id]:
        data = await state.get_data()
        items: list[dict[str, Any]] = data.get("selected_items", [])
        if not items:
            await callback.answer("⚠️ Ты пока ничего не выбрал.", show_alert=True)
            return
        
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        
        keep_ids = {callback.message.message_id}
        await delete_previous_bot_messages(callback.message.chat.id, callback.bot, keep_ids)
        
        wait_msg = await callback.message.answer("🧠 Думаю, собираю JSON и ищу совпадения в базе...")
        try:
            photo_path = Path(data["photo_path"])
            json_str = await generate_recommendation_json(photo_path, items)
            if not json_str:
                await send_error(
                    callback.message.chat.id,
                    callback.message.bot,
                    "⚠️ Не смог получить корректный JSON. Попробуй другое фото или другие категории.",
                )
                await safe_delete_message(callback.bot, callback.message.chat.id, wait_msg.message_id)
                return
            
            if DEBUG_JSON:
                json_msg = await callback.message.answer(
                    f"<b>JSON от модели:</b>\n<code>{json_str}</code>", parse_mode=ParseMode.HTML
                )
                keep_ids.add(json_msg.message_id)
            
            matches = await search_matches_async(json_str)
            if not matches:
                await send_info(callback.message.chat.id, callback.message.bot,
                                "Ничего не нашлось. Попробуй изменить запрос.")
                await safe_delete_message(callback.bot, callback.message.chat.id, wait_msg.message_id)
                return
            
            if DEBUG_JSON:
                await safe_delete_message(callback.bot, callback.message.chat.id, json_msg.message_id)
            
            sent_messages = []
            results_found = False
            
            for m in matches:
                caption = (
                    f"🎯 <b>{m['query_category']}</b>\n"
                    f"🔎 Сходство: <b>{m['match_score']}</b>\n"
                    f"🆔 {m['set_id']}_{m['index']}\n"
                    f"📝 {m['original_desc']}"
                )
                if m.get("photo") and Path(m["photo"]).exists():
                    msg = await callback.message.answer_photo(
                        photo=FSInputFile(m["photo"]),
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                    )
                    results_found = True
                else:
                    msg = await callback.message.answer(caption + "\n⚠️ Фото не найдено.", parse_mode=ParseMode.HTML)
                sent_messages.append(msg.message_id)
            
            await safe_delete_message(callback.bot, callback.message.chat.id, wait_msg.message_id)
            
            if results_found:
                all_keep_ids = {callback.message.message_id, *sent_messages}
                await delete_previous_bot_messages(callback.message.chat.id, callback.bot, all_keep_ids)
                
                restart_msg = await callback.message.answer(
                    "✅ Анализ завершен!\n\n"
                    "📸 <b>Хочешь проанализировать другой образ?</b>\n"
                    "Просто отправь мне новое фото!",
                    parse_mode=ParseMode.HTML
                )
                
                await state.clear()
                await state.set_state(ClothingStates.waiting_for_photo)
                
                sent_messages.append(restart_msg.message_id)
                
            else:
                await callback.message.answer(
                    "❌ По заданным критериям ничего не найдено.\n\n"
                    "📸 Попробуй с другим фото или измени категории поиска.\n"
                    "Отправь новое фото чтобы начать сначала.",
                    parse_mode=ParseMode.HTML
                )
                
                await state.clear()
                await state.set_state(ClothingStates.waiting_for_photo)
            
        except Exception:
            logger.exception("analyze_selected error")
            await send_error(
                callback.message.chat.id,
                callback.message.bot,
                "❌ На пайплайне произошла ошибка. Попробуй ещё раз или покажи логи администратору.\n\n"
                "📸 Отправь новое фото чтобы начать сначала.",
            )
            await state.clear()
            await state.set_state(ClothingStates.waiting_for_photo)
        finally:
            await callback.answer()


async def main() -> None:
    global SEARCH_ENGINE

    logger.info("Init FashionSearchEngine...")
    SEARCH_ENGINE = FashionSearchEngine(base_path=FASHION_DATA_PATH, device=FASHION_DEVICE)

    logger.info("Init SGLang Runtime for Qwen3-VL-4B-Thinking...")
    try:
        ckpt = "Qwen/Qwen3-VL-4B-Thinking"
        runtime = sgl.Runtime(
            model_path=ckpt,
            tp_size=1,
            dtype="float16",
            attention_backend="flashinfer",
            enable_torch_compile=False,
            disable_cuda_graph=False,
            mem_fraction_static=0.7,
            context_length=10240,
            chunked_prefill_size=512,
        )
        sgl.set_default_backend(runtime)
        logger.info("SGLang started: %s", ckpt)
    except Exception:
        logger.exception("FATAL: cannot start SGLang")
        return

    token = os.getenv("BOT_TOKEN")
    if not token and os.path.exists("token.txt"):
        with open("token.txt", "r", encoding="utf-8") as f:
            token = f.read().strip()
    if not token:
        raise RuntimeError("Нет BOT_TOKEN (env) и нет token.txt")

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("Bot polling started.")
    try:
        await dp.start_polling(bot)
    finally:
        if runtime:
            logger.info("Shutdown SGLang runtime...")
            runtime.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass