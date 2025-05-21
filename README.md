### Описание проекта: Генерация изображений из текста для иллюстрации персонализированной истории

## 🛠️ Usage
Установка окружения
```
bash install_scripts/runpod_install_oneactor.sh
bash install_scripts/runpod_install_dreamsim.sh
```
Пример запуска модели для промптов ```config/prompt-adventurer.yaml```, результат будет сохранен в директорию ```experiments/adventurer```.
```
bash test_scripts/script.sh config/prompt-adventurer.yaml adventurer
```
<div align="center">
    <img src="gallery/adventurer_a_city_as_background.jpg", width="600">
    <br><br><br>
    <img src="gallery/gentelman_eating_a_burger.jpg", width="600">
    <br><br><br>
    <img src="gallery/waiting_at_a_bus_stop.jpg", width="600">
</div>

## 🤖 TelegramBot
```
python one_actor_bot.py
```
<div align="center">
    <img src="gallery/tg_bot.png", width="600">
</div>


## 🔗 Links
**OneActor** https://github.com/JoHnneyWang/OneActor

**ConsiStory** https://github.com/NVlabs/consistory
