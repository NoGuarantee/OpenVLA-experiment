# Эксперимент OpenVLA: сдвиг объекта по X

Проверка, как меняется предсказание OpenVLA при горизонтальном сдвиге объекта на кадре.

## Файлы

| Путь | Назначение |
|------|------------|
| `experiment.ipynb` | Загрузка модели, sweep по `dx`, графики |
| `image_shift.py` | Композит: чистый фон + объект по маске |
| `images/image_original.jpg` | Исходный кадр с объектом |
| `images/image_mask.png` | Ч/б маска объекта |
| `images/image_background.png` | Фон без объекта (inpaint) |
| `setup_env.sh` | Создание `.venv` и зависимостей |
| `requirements.txt` | Python-пакеты (без torch) |

## Установка

```bash
bash setup_env.sh
source .venv/bin/activate
```

Модель `openvla/openvla-7b` скачивается с Hugging Face при первом запуске ноутбука.

## Запуск

```bash
jupyter notebook experiment.ipynb
```

Результаты sweep (опционально) сохраняются в `outputs/shift_sweep/` — каталог в `.gitignore`.

## Пайплайн

1. Маска → bbox и вырез объекта с альфой.
2. Для каждого `dx` объект вставляется на `image_background.png`.
3. `predict_action` с фиксированным промптом; сравнение 7 компонент действия на графиках.
