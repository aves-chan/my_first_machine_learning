## Домашняя работа №1

Решение задачи мультилейбл классификации на примере определения типа спутникового снимка лесов Амазонки.

### Датасет

Включает 40479 тайлов спутниковых снимков в формате `jpg` и 17 типов изображений. Более подробная информация в [тетрадке](notebooks/EDA.ipynb). 

Скачать датасет (он окажется в папке dataset):

```bash
make download_dataset
```

### Подготовка окружения

1. Создание и активация окружения
    ```bash
    python3 -m venv venv
    . venv/bin/activate
    ```

2. Установка библиотек
   ```
    make install
   ```
   
3. Запуск линтеров
   ```
   make lint
   ``` 

4. Логи лучшего эксперимента в ClearML

- https://app.clear.ml/projects/ad24a8e7ed7744c8bf6216f8b8c710cf/experiments/c13b215c07ec48f587d3ec8f5f1cf970/output/execution
- https://app.clear.ml/projects/ad24a8e7ed7744c8bf6216f8b8c710cf/experiments/ad36b4c4865a46bc83bc64d8d04e434c/output/execution
- https://app.clear.ml/projects/ad24a8e7ed7744c8bf6216f8b8c710cf/experiments/f9a15c5e1c9b4b9999e4ff8b1163ebc4/output/execution
- https://app.clear.ml/projects/ad24a8e7ed7744c8bf6216f8b8c710cf/experiments/df4a05465fdb4a4582de97f760fc0a30/output/execution


5. Настраиваем [config.yaml](configs/config.yaml) под себя.


### Обучение

Запуск тренировки:

```bash
make train
```

### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb).


All fixed!