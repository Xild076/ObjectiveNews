# Objective News
By: Xild076 (Harry Yin)

## Project Description
This project was created to help combat misinformation in news. The code gathers news articles from across the web about a certain topic, divides the information into groups, objectifies and summarizes all the gathered information, and calculates the reliability of the information to help readers determine the truth of certain facts to a higher degree. Each of the tools: summarization, objectification, and grouping are all avaliable separately.

## Prerequisites
- Install dependancies by running `pip install -r requirements.txt` or `pip3 install -r requirements.txt`

## Usage
### Installation
- Run `streamlit run src/app.py` or `python3 -m streamlit run src/app.py`
- Import from article_analysis.py, grouping.py, objectify_text.py, summarizer.py, synonym.py, scraper.py, or util.py for any of the tools
### Vercel Deployment (API)
- This repo now includes a FastAPI server under `api/index.py` and `vercel.json` for deployment on Vercel.
- Deploy steps:
    1. Install the Vercel CLI: `npm i -g vercel`
    2. Login: `vercel login`
    3. From the repo root, deploy: `vercel`
- After deploy, call endpoints:
    - Health: `GET /api/health`
    - Cluster sentences: `POST /api/cluster` with `{ "sentences": ["...", "..."] }`
    - Cluster texts: `POST /api/cluster-texts` with a list of `{text, source?, author?, date?}`
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
