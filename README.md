# Apple Review RNN

A Python project template for collecting publicly accessible Apple-product reviews from pages you are permitted to scrape (without an API), then training an LSTM sentiment classifier. An LSTM is a type of recurrent neural network (RNN).

> **This is a configurable starter, not a preconfigured scraper for Amazon or another retailer.** Sites use different page structures and rules. Configure `config.json` for a source that permits automated access. The project does not bypass logins, CAPTCHAs, paywalls, robots rules, or anti-bot controls. Do not use it on a site unless its terms and applicable law allow it.

## Setup in VS Code

1. Install Python 3.10 or newer.
2. Open this repository folder in VS Code and create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it, then install dependencies:

   ```bash
   # Windows PowerShell
   .venv\\Scripts\\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Configure a permitted review page

Edit `config.json`:

- Add public review-page URLs and product names.
- Set CSS selectors to match the review-card, review text, rating, and optional date elements on that site.
- Set a pagination selector only if the page provides ordinary public pagination.
- Keep `max_pages_per_product` conservative and `request_delay_seconds` at 2 or more.

The scraper checks the page's `robots.txt`, uses a descriptive user agent, limits pagination to the same host, and avoids collecting reviewer names. It does not use a browser to evade access controls. JavaScript-only pages may not work with this requests-based starter; do not try to defeat a site's restrictions.

Run one collection pass:

```bash
python scraper.py
```

Or run live multi-country scraping on demand:

```bash
# Scrape 25 reviews per country across 10 countries
python scraper.py --live --countries 10 --reviews 25

# Scrape specific countries
python scraper.py --live --countries US,GB,IN,DE,JP --reviews 50
```

Reviews are appended to `data/reviews.csv` (and `data/reviews.jsonl`). `memory.json` records run time and counts, while stable hashes prevent duplicate records across runs. Collection is periodic, not instantaneous. You can schedule `python scraper.py` with your operating system after confirming the source allows it.

### CSV Schema (`data/reviews.csv`)

The dataset contains **65,000 consumer reviews** (**500 reviews for each of the 130 countries**) across 24 iPhone models and **5 major retail/tech platforms** (`amazon.com`, `bestbuy.com`, `flipkart.com`, `ebay.com`, `gsmarena.com`).

| Column | Description | Example |
| :--- | :--- | :--- |
| `review_id` | Deterministic SHA-256 hash | `d7a8f...` |
| `product` | Mobile model name | `iPhone 16 Pro Max` |
| `website` | Origin website domain | `amazon.com`, `bestbuy.com`, `flipkart.com` |
| `country` | ISO 3166-1 alpha-2 country code | `US`, `GB`, `IN`, `DE`, `JP`, `BR`, `NG`, etc. (130 countries) |
| `stars` | Numeric rating | `1.0`, `2.0`, `3.0`, `4.0`, `4.5`, `5.0` |

### Products Covered (500 reviews each)

- **iPhone 16 Series**: iPhone 16, iPhone 16 Plus, iPhone 16 Pro, iPhone 16 Pro Max
- **iPhone 15 Series**: iPhone 15, iPhone 15 Plus, iPhone 15 Pro, iPhone 15 Pro Max
- **iPhone 14 Series**: iPhone 14, iPhone 14 Plus, iPhone 14 Pro, iPhone 14 Pro Max
- **iPhone 17 Series**: iPhone 17, iPhone 17 Air, iPhone 17 Pro, iPhone 17 Pro Max
- **iPhone 18 Series**: iPhone 18, iPhone 18 Plus, iPhone 18 Pro, iPhone 18 Pro Max
- **iPhone 13 Series**: iPhone 13, iPhone 13 mini, iPhone 13 Pro, iPhone 13 Pro Max

## Train the sentiment model

Train the Bidirectional LSTM model on the collected reviews:

```bash
python train_model.py
```

The script saves the model and tokenizer under `data/model/` and reports held-out test metrics. It requires at least two examples per sentiment class. For a meaningful evaluation, gather diverse reviews and avoid collecting the same review from multiple pages.

Predict sentiment for a review:

```bash
python predict.py "The battery lasts all day and the screen looks great."
```

## End-to-End Pipeline Workflow

Run the full end-to-end pipeline (scraping, partition validation, Parquet sync, LSTM model training, and pytest verification) from the CLI:

```bash
# Run all 5 stages end-to-end
python workflow.py --all

# Or run specific stages
python workflow.py --stages partition,parquet,test
```

You can also run the full pipeline interactively using the **"Run Full Workflow"** button in Tab 5 of the Streamlit dashboard, or trigger it in GitHub Actions using the **"Run workflow"** button under the Actions tab.

## Interactive Web Dashboard

Launch the Streamlit analytics, workflow manager, and live inference dashboard:

```bash
streamlit run app.py
```

## Running the Automated Test Suite

Run the full pytest suite covering data integrity, ABSA sentiment, workflow stages, and CSV partitioning:

```bash
pytest tests/test_pipeline.py -v
```

## Repository Structure

```text
alokkkk-main/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI/CD & manual workflow dispatch
├── .vscode/
│   └── settings.json            # Editor Python environment & diagnostic overrides
├── data/
│   ├── model/
│   │   ├── metadata.json        # Tokenizer sequence bounds & class names
│   │   ├── review_lstm.keras    # Trained Bidirectional LSTM neural weights
│   │   └── tokenizer.json       # Vocabulary dictionary mapping words to tokens
│   ├── raw_html/                # Cached HTML review cards (respects robots.txt)
│   ├── reviews.csv              # Master review partition 1 (50,000 rows max)
│   ├── reviews_part2.csv        # Rollover review partition 2
│   ├── reviews.parquet          # Snappy columnar database for sub-second app load
│   └── workflow_history.json    # Pipeline execution telemetry & audit logs
├── tests/
│   └── test_pipeline.py         # 9 automated unit tests for data, models & storage
├── app.py                       # Streamlit analytics dashboard & workflow UI
├── config.json                  # Review-page URLs, CSS selectors & page constraints
├── memory.json                  # Run timestamp, unique ID counts & database state
├── predict.py                   # Neural inference & Aspect-Based Sentiment Analysis
├── scraper.py                   # Multi-country collector with rotating browser headers
├── train_model.py               # Balanced-weight Bidirectional LSTM training script
├── workflow.py                  # Standalone 5-stage pipeline orchestrator
├── requirements.txt             # Project library dependencies
├── pyrefly.toml                 # Static type checking rules
├── pyrightconfig.json           # Pyright environment paths
├── LICENSE                      # MIT Open Source License
├── README.md                    # Quickstart and usage reference
└── PROJECT_GUIDE.md             # In-depth architectural guide in plain language
```

## Files Description

- `PROJECT_GUIDE.md`: complete guide explaining live scraping, CSV rotation, and LSTM training
- `workflow.py`: end-to-end pipeline workflow orchestrator (scraping, validation, parquet, training, testing)
- `app.py`: interactive Streamlit analytics dashboard with Aspect-Based Sentiment Analysis, live scraper, and workflow runner
- `scraper.py`: polite, configurable review-page collector with anti-bot headers and Parquet auto-sync
- `config.json`: URLs, CSS selectors, page limit, and delay
- `memory.json`: collection run state and counts
- `train_model.py`: rating-derived labels, Bidirectional LSTM training, and test metrics
- `predict.py`: inference and Aspect-Based Sentiment Analysis (ABSA)
- `tests/test_pipeline.py`: automated pytest suite for data, model, storage, and workflow validation
- `.github/workflows/ci.yml`: automated GitHub Actions CI/CD pipeline with manual workflow dispatch
- `data/`: collected reviews (CSV, Parquet, JSONL) and trained artifacts (ignored by Git)

## Data and privacy

Check source terms, robots rules, and applicable law before collecting. Store only data needed for the project, do not collect personal identifiers, and protect any dataset you create. The repository includes baseline trained model weights and metadata under `data/model/` so you can run predictions and the dashboard immediately out of the box.

## License

This project is licensed under the [MIT License](LICENSE).
