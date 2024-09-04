# Analyzing Car Reviews with LLMs

## Project Overview

This project utilizes advanced language models (LLMs) to analyze car reviews, performing tasks like sentiment analysis, translation, question answering, and summarization. By applying state-of-the-art NLP techniques, we aim to extract insights from car reviews, enhancing customer understanding and improving decision-making.

Key Aspects:

Data Preprocessing: Handling and preparing car reviews data for analysis using LLMs.
Sentiment Analysis: Classifying reviews as positive or negative based on their content.
Translation: Translating car reviews from English to Spanish.
Question Answering: Extracting answers from reviews based on specific questions.
Summarization: Summarizing long reviews to provide concise insights.
Containerization: Docker is used to encapsulate the project environment for easy replication.

Features:

Sentiment Analysis: Using LLMs to classify the sentiment of car reviews.
Translation: Translating English car reviews to Spanish using NLP models.
Question Answering: Extracting specific information from reviews using LLMs.
Summarization: Summarizing lengthy car reviews for quick insights.
Containerized Environment: Dockerized project to ensure consistency across different environments.

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




