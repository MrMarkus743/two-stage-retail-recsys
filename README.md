# Two-Stage Retail Recommendation System (`two-stage-retail-recsys`)

Высокопроизводительный двухстадийный рекомендательный сервис (Candidate Retrieval + Ranking) для персонализации корзины в e-commerce на базе датасета Instacart (32+ млн записей).

---

## 🛠 Tech Stack

- **Язык и среда:** Python 3.10+
- **Big Data & Out-of-Core ETL:** Polars, Apache Parquet
- **Алгоритмическое ядро и структуры данных:** SciPy (CSR Sparse Matrix), NumPy, Binary Min-Heap ($O(N \log K)$)
- **Сервисный слой:** FastAPI, Pydantic v2, Uvicorn *(Sprint 3)*
- **Кэширование и хранение:** Redis (паттерн Cache-Aside) *(Sprint 3)*
- **Контейнеризация и оркестрация:** Docker, Docker Compose *(Sprint 3)*
- **Инструменты качества и тестирование:** Ruff, Pytest, GitHub Actions CI

---

## 📂 Repository Structure

```text
two-stage-retail-recsys/
├── benchmarks/             # Скрипты нагрузочного тестирования и замера SLA
├── data/
│   └── sample/             # Легковесные микросрезы данных (< 1000 строк) для тестов и CI
├── docs/
│   └── adr/                # Архитектурные решения (Architectural Decision Records)
├── src/
│   ├── api/                # Эндпоинты FastAPI, роутинг и схемы валидации Pydantic
│   ├── core/               # Алгоритмическое ядро: куча (Min-Heap), метрики сходства
│   ├── infrastructure/     # Инфраструктурные клиенты (Redis, кэширование)
│   └── pipeline/           # Конвейеры: ETL (Polars), генератор матрицы CSR, каталог
├── tests/
│   ├── integration/        # Интеграционные тесты API и кэша
│   └── unit/               # Модульные тесты алгоритмов и конвейеров
├── .gitignore              # Правила исключения сырых датасетов, дампов и кэшей
├── LICENSE                 # Лицензия проекта
├── README.md               # Документация и витрина репозитория
└── requirements.txt        # Зафиксированные версии базовых зависимостей
```

---

## Quick Start

### 1. Клонирование и переход в проект
```bash
git clone https://github.com/MrMarkus743/two-stage-retail-recsys.git
cd two-stage-retail-recsys
```

### 2. Настройка виртуального окружения

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```