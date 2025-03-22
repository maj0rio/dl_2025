# Loan prediction

## Эксперимент 1. Простая модель

### Параметры для обучения

```yaml
lr: 0.01
hidden_size: 32
num_epochs: 10
batch_size: 128
weight_decay: 0.0001
seed: 1611
```

### Архитектура модели

Эмбеддинги для всех категориальных фичей + отображение всех числовых в hidden_size

- Linear (hidden size в hidden size * 4)
- ReLU
- Linear (hidden size * 4 в hidden size)
- Linear(hidden size в 1)

### Результаты

В папке [experimets](experiments) с префиксом 1
`ROC_AUC val: 0.8630`
Виден потенциал модели, есть понимание как улучшать качество


## Эксперимент 2. Увеличиваем сложность

### Параметры для обучения

```yaml
lr: 0.01
hidden_size: 128
num_epochs: 10
batch_size: 128
weight_decay: 0.0001
seed: 1611
```

### Архитектура модели

Эмбеддинги для всех категориальных фичей + отображение всех числовых в hidden_size

- Linear (hidden size в hidden size * 4)
- ReLU
- Linear (hidden size * 4 в hidden size)

- Linear (hidden size в hidden size * 4)
- ReLU
- Linear (hidden size * 4 в hidden size)

- Linear (hidden size в hidden size * 4)
- ReLU
- Linear (hidden size * 4 в hidden size)

- Linear(hidden size в 1)

### Результаты

В папке [experimets](experiments) с префиксом 2_
`ROC_AUC val: 0.8998`

Стало получше. Видно, что при увеличении кол-ва эпох метрика повышается



## Эксперимент 3. Skip Connections, Batch Norms


### Параметры для обучения

Как в пункте 2

### Архитектура модели

Эмбеддинги для всех категориальных фичей + отображение всех числовых в hidden_size. Плюс реализация Skip Connections для 3 блоков. Добавляем Batch Norm к каждому блоку

### Результаты

В папке [experimets](experiments) с префиксом 3_
`ROC_AUC val: 0.9276`

Очень хороший результат


## Эксперимент 4. Dropout

### Параметры для обучения

Как в пункте 2 +
```yaml
drop_p: 0.1
```
### Архитектура модели

Добавляем еще дропаут, чтобы модель не переобучалась на конкретных _нейронах_


### Результаты

В папке [experimets](experiments) с префиксом 4_
`ROC_AUC val: 0.9234`

Метрика примерно такая же. Теперь модель аккуратно основывается на всех фичах и не перееобучается



## Эксперимент 5. Подгон гиперпараметров

### Параметры для обучения

Как в пункте 2 +
```yaml
  drop_p: 0.1
  lr: 0.01
  hidden_size: 128
  num_epochs: 15
  batch_size: 128
  weight_decay: 0.001
  seed: 1611
```
### Архитектура модели

То же самое

### Результаты

Поигравшись чуть чуть с параметрами, можно получить следующие результаты. Что вполне может устроить

В папке [experimets](experiments) с префиксом 5_
`ROC_AUC val: 0.9254`

