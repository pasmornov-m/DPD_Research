# Исследование методов цифровой коррекции (DPD)

В этом репозитории представлен процесс моделирования и анализа качества **Digital Predistortion (DPD)** с использованием различных моделей машинного обучения.

## Реализованные модели:

* **GMP** — *Generalized Memory Polynomial* (полиномиальная модель с учётом памяти)
* **GRU** — *Gated Recurrent Unit* (RNN)
* **LSTM** — *Long Short-Term Memory* (RNN)
* **TCN** — *Temporal Convolutional Network* (CNN)
* **DenseNet** — *Densely Connected Convolutional Network* (CNN)
* **ESN** — *Echo State Network* (Reservoir Computing)

## Архитектуры обучения DPD:

После выбора модели реализуются три подхода к обучению цифровых корректоров:

* **DLA** — *Direct Learning Architecture* (прямая архитектура обучения)
* **ILA** — *Indirect Learning Architecture* (обратная архитектура обучения)
* **ILC** — *Iterative Learning Control* (итеративный контроль обучения)

## Метрики и визуализация:

Обучение проводится с использованием выбранной модели и выбранной архитектуры.
Для сравнения качества работы используются:

* Численные метрики: **NMSE** (Normalized Mean Squared Error), **ACPR** (Adjacent Channel Power Ratio)
* Визуализация: **PSD** (Power Spectral Density)

## Цели исследования:

* Провести симуляционное моделирование различных моделей и архитектур DPD
* Сравнить эффективность DLA, ILA и ILC в различных условиях (уровень шума, параметры модели)
* Исследовать влияние отношения сигнал/шум на эффективность работы корректоров


---


# Research on Digital Predistortion (DPD) Methods

This repository presents the process of modeling and analyzing the performance of **Digital Predistortion (DPD)** using various machine learning models.

## Implemented Models:

* **GMP** — *Generalized Memory Polynomial* (polynomial model with memory effects)
* **GRU** — *Gated Recurrent Unit* (RNN)
* **LSTM** — *Long Short-Term Memory* (RNN)
* **TCN** — *Temporal Convolutional Network* (CNN)
* **DenseNet** — *Densely Connected Convolutional Network* (CNN)
* **ESN** — *Echo State Network* (Reservoir Computing)

## DPD Training Architectures:

After selecting a model, three approaches to training digital predistorters are implemented:

* **DLA** — *Direct Learning Architecture*
* **ILA** — *Indirect Learning Architecture*
* **ILC** — *Iterative Learning Control*

## Metrics and Visualization:

Training is performed using the selected model and architecture.
The following evaluation methods are used:

* Numerical metrics: **NMSE** (Normalized Mean Squared Error), **ACPR** (Adjacent Channel Power Ratio)
* Visualization: **PSD** (Power Spectral Density)

## Research Objectives:

* Perform simulation-based modeling of different DPD models and architectures
* Compare the effectiveness of DLA, ILA, and ILC under various conditions (noise level, model parameters)
* Investigate the impact of the signal-to-noise ratio on predistorter performance

