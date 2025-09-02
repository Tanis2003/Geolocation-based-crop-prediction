# Crop Yield Recommendation System 🌱

An intelligent web application built with Django that predicts optimal crop yields by analyzing soil nutrient data and real-time weather conditions.

-----

## 📋 Overview

This project is a data-driven system designed to assist farmers in making informed decisions. By inputting a location, the application retrieves soil nutrient data, fetches current weather information from an external API, and utilizes a machine learning model to predict the most suitable crops for that region, ultimately aiming to enhance agricultural productivity.

-----

## 🔬 Methodology

The core logic of the application follows a multi-step process to generate a recommendation:

1.  **Data Extraction**: The system queries a local **MySQL database** to retrieve essential soil data for a specified city. This includes the levels of key macronutrients (like Nitrogen, Phosphorus, Potassium) and micronutrients.

2.  **Weather API Integration**: It makes a real-time **API call** to an external weather service to fetch current environmental conditions such as temperature, humidity, and rainfall for the given location.

3.  **Prediction Model**: The collected soil and weather data are fed into a **machine learning model** (trained using scikit-learn). This model performs calculations to analyze the parameters and predicts a list of crops that would thrive in those specific conditions.

4.  **User Interface**: The final recommendations are displayed to the user through a clean and interactive web interface built with Django.

-----

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

  * Python 3.8+
  * pip
  * A local MySQL server instance

### Local Installation and Setup

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and Activate a Virtual Environment**
    It's highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it
    source venv/bin/activate
    ```

    On Windows, activation is `venv\Scripts\activate`.

3.  **Install Required Libraries**
    Create a `requirements.txt` file in the root of your project directory and add the following libraries:

    ```
    Django
    pymysql
    pandas
    scikit-learn
    requests
    geopy
    matplotlib
    ```

    Now, install them all with one command:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Database Setup**

      * Make sure your MySQL server is running.
      * Create a new database named `cropyield`.
      * **Important**: You may need to import the database schema or data if you have a `.sql` dump file.

5.  **Run Django Migrations**
    This will set up the necessary tables in your database.

    ```bash
    python manage.py migrate
    ```

6.  **Run the Development Server**
    You're all set\! Start the server with this command:

    ```bash
    python manage.py runserver
    ```

    The project will now be running at `http://127.0.0.1:8000/`.

-----

## ⚠️ Note for Windows Users

This project was developed natively on **macOS**. If you are running this on a Windows machine, you may encounter hardcoded file paths (e.g., for datasets or static files). Please search the codebase for paths like `'Recommendationsystem/dataset/...'` and update them to the appropriate Windows format (e.g., `'Recommendationsystem\\dataset\\...'`).
