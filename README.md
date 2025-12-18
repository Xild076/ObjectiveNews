# Objective News
By: Xild076 (Harry Yin)

## Project Description
This project was created to help combat misinformation in news. The code gathers news articles from across the web about a certain topic, divides the information into groups, objectifies and summarizes all the gathered information, and calculates the reliability of the information to help readers determine the truth of certain facts to a higher degree. Each of the tools: summarization, objectification, and grouping are all avaliable separately.

## Prerequisites
- Install dependancies by running `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

## Usage
### Streamlit research studio (local)
- Run `streamlit run src/app.py` or `python3 -m streamlit run src/app.py`
- Import from article_analysis.py, grouping.py, objectify_text.py, summarizer.py, synonym.py, scraper.py, or util.py for any of the tools

- The Next.js source now lives in `frontend/` to keep it isolated from the Streamlit codebase.
- Install Node dependencies once: `cd frontend && npm install`
- Local development (in one terminal run the API, in another the UI):
    ```bash
    /Users/harry/Documents/Python_Projects/ObjectiveNews/.venv/bin/python -m uvicorn api.index:app --reload --port 8000
    cd frontend && NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
    ```
- Production build: `cd frontend && npm run build` (used automatically by Vercel)
- The deployed site calls the Python API through `/api/*` thanks to `vercel.json` rewrites.

### FastAPI endpoints / Vercel deployment
- FastAPI server lives in `api/index.py` and is exposed on Vercel via `api/index.py`.
- Deploy steps:
    1. Install the Vercel CLI: `npm i -g vercel`
    2. Login: `vercel login`
    3. From the repo root, deploy: `vercel` (the CLI runs `npm run build` and packages the Python function with `requirements.txt`).
- After deploy, call endpoints:
    - Health: `GET /api/health`
    - Cluster sentences: `POST /api/cluster` with `{ "sentences": ["...", "..."] }`
    - Cluster texts: `POST /api/cluster-texts` with a list of `{text, source?, author?, date?}`
    - Group a long-form article: `POST /api/group-article` with `{text, source?, author?, date?}`
- Streamlit remains available for local use: `streamlit run src/app.py`.
### Website
- See [Objective News](https://objectivenews.streamlit.app/) for the applications run on streamlit.
    - Just as a heads up, due to memory contraints, the app will be somewhat prone to crashes! I am currently working to improve the situation.
### Necessary Specs:
- At least 2 core CPU
- No GPU needed
- 2.5 GB Memory

## Contributing
- Issue Tracker: [Issues - Xild/ObjectifyNews](https://github.com/Xild076/ObjectiveNews/issues)

## Directories:
- `data/`: Stores data about bias
 - `images/`: Stores images for the website page
- `src/`: Source code
    - `pages`: Store website page information
- `requirements.txt`: Stores requirements
- `paper/`: Stores paper

## Paper (informal);
[Read the paper](./paper/paper.md)

## Contact
- Email: harry.d.yin.gpc@gmail.com
