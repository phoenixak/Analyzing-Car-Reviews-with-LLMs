# Recipe Site Traffic Analysis & Prediction

This project uses advanced language models to analyze car reviews. It includes tasks such as sentiment analysis, translation, question answering, and summarization. The project is modularized, and integrates MLflow for experiment tracking and Docker for containerization.

## Project Overview

The objective of this project is to analyze car reviews using various language models and develop functionalities such as sentiment classification and text summarization. By leveraging state-of-the-art language models and applying them to car reviews, we aim to provide insights into customer opinions, improve user interactions, and enhance the overall review analysis process.

Key Aspects:
Data Preprocessing: Cleaning and transforming raw traffic data for analysis.
Exploratory Data Analysis (EDA): Visualizing and understanding traffic trends.
Model Development: Training machine learning models to predict traffic.
Model Evaluation: Assessing model performance using metrics like MAE, RMSE, etc.
Containerization: Docker is used to encapsulate the project environment.
Experiment Tracking: MLFlow is integrated for tracking experiments and model versions.

Features
Traffic Analysis: Detailed analysis of traffic sources, user behavior, and site performance.
Predictive Modeling: Implementation of regression models to predict future site traffic.
Visualization: Interactive and static visualizations to understand traffic patterns.
Containerized Environment: Easily reproducible environment using Docker.
Experiment Tracking: MLFlow is used to track different model versions and experiments.

### Prerequisites

To run this project, you'll need to have the following software installed:

- Python 3.9 or higher
- pip (Python package installer)

You can install Python and pip by following the instructions [here](https://www.python.org/downloads/).

### Installing

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/phoenixak/Analyzing-Car-Reviews-with-LLMs.git
   cd Analyzing-Car-Reviews-with-LLMs

2. **Create a virtual environment:**

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
Or 

1. **Build the Docker image:**

   ```bash
   docker build -t Analyzing-Car-Reviews-with-LLMs .
  
   Run the Docker container:
2. **Run the Docker container:**

   ```bash
   Copy code
   docker run -v "$(pwd)/dataset":/app/dataset -it Analyzing-Car-Reviews-with-LLMs

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any feature requests or bugs.




