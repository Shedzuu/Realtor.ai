# Project Title

## Introduction
This project is a real estate recommendation system that uses machine learning to match user preferences with available properties.

## Problem Statement
Finding the right property that matches user preferences can be time-consuming and inefficient. This project aims to streamline the process by providing personalized recommendations based on user input.

## Objectives
- To simplify the property search process.
- To provide accurate and personalized property recommendations.
- To enhance user experience with a data-driven approach.

## Technology Stack
- **Frontend**: HTML, CSS, JS
- **Backend**: Python, Flask, DeepSeek API
- **Database**: CSV-based dataset
- **Others**: Scikit-learn, Pandas

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Shedzuu/Realtor.ai.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Realtor.ai
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide
### Option 1: Run with Web Interface (Recommended)
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```
3. Enter your property preferences in the provided interface.
4. The system will generate a list of recommended properties based on your input.

### Option 2: Run via Python Script
To run the application directly without web interface:
```bash
python src/model.py
```




## Known Issues / Limitations
- Limited to the dataset provided in `data/apartments.csv`.
- Requires an active internet connection for DeepSeek API.

## References
- DeepSeek API documentation
- Scikit-learn documentation
- Pandas documentation
- Flask documentation

## Team Members
- Kuanysh Sembay, 230103202, Group 20-P
- Sagyngaliyeva Aruzhan, 220103087, Group 20-P
- Akkuzov Arman, 220103145, Group 20-P
- Zhenis Kuandyk, 220103026, Group 20-P
- SaIdulla Ramazan, 220103054, Group 20-P