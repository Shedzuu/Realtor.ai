# Realtor.ai Project

## Overview
Realtor.ai is a project designed to scrape apartment listings, process real estate data, and provide recommendations based on user input. The project leverages Python for data scraping, processing, and machine learning, and includes integration with OpenAI's API for advanced data extraction.

## Files in the Project

### 1. `model.ipynb`
This Jupyter Notebook contains the following functionalities:
- **Data Loading**: Reads apartment data from `apartments.csv`.
- **OpenAI Integration**: Uses OpenAI's API to extract structured real estate details from user input.
- **Similarity Calculation**: Implements a function to calculate similarity scores between user preferences and available apartments using TF-IDF and cosine similarity.
- **Sorting Results**: Outputs apartments sorted by their similarity scores.

### 2. `apartments_scrap.ipynb`
This Jupyter Notebook is responsible for:
- **Web Scraping**: Scrapes apartment listings from the `krisha.kz` website using BeautifulSoup.
- **Data Extraction**: Extracts details such as price, location, floor, area, and additional features of apartments.
- **Data Storage**: Saves the scraped data into a structured format (e.g., `apartments.csv`).

### 3. `apartments.csv`
This file contains the scraped apartment data in a tabular format. Each row represents an apartment listing with columns for features such as:
- URL
- Number of rooms
- Address
- Price
- City/Region
- Floor
- Area
- Additional features (e.g., kitchen type, security, furniture details, etc.)

## How to Use

1. **Scrape Data**:
   - Run `apartments_scrap.ipynb` to scrape apartment listings and save them to `apartments.csv`.

2. **Analyze Data**:
   - Open `model.ipynb` to load the scraped data, process user input, and calculate similarity scores.

3. **API Integration**:
   - Ensure your OpenAI API key is set as an environment variable (`OPENAI_API_KEY`) before running the notebooks.

## Requirements
- Python 3.8+
- Jupyter Notebook
- Libraries:
  - `pandas`
  - `BeautifulSoup4`
  - `requests`
  - `scikit-learn`
  - `openai`

## Setup
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## License
This project is licensed under the MIT License.