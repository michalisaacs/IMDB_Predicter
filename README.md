[README (3).md](https://github.com/user-attachments/files/29014496/README.3.md)
# IMDb Movie Rating Predictor 🎬

## 1. Project Description
This project delivers a machine learning solution designed to predict IMDb movie ratings on a scale from 1 to 10. Built around a **Random Forest Regressor**, the model processes key cinematic features, including a custom actor reputation metric calculated over a 10-year historical window, director reputation metrics, movie runtime classification (mapped into discrete size bins), principal genres, and the original language of the film. The system includes a backend REST API developed with Flask and a responsive web interface allowing users to input raw movie metadata and receive instantaneous rating predictions.

---

## 2. Installation Instructions
To set up and run this project locally, ensure you have Python installed and execute the following commands in your terminal (Git Bash or Command Prompt):

1. **Navigate to the project directory:**
   ```bash
   cd /path/to/your/IMDB_Predicter
