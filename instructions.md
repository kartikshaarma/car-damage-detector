Car Damage Detection Project: Setup and Execution Guide
Step 1: Project Setup
Create a Project Folder: Create a new folder named car_damage_detection.

Add Files: Place the three files you just created (dataset.yaml, train.py, app.py) inside this folder.

Download Dataset: Download the CarsDD Dataset and unzip it. Place the resulting CarsDD folder inside your car_damage_detection folder.

Install Libraries: Open your terminal, navigate into your project folder, and run:

pip install ultralytics streamlit opencv-python Pillow

Step 2: Train the Model
In your terminal (make sure you are inside the car_damage_detection folder), run the training script:

python train.py

Wait for the training to finish. It will create a runs folder containing your trained model.

Step 3: Run the Web App
Once training is done, run the Streamlit app from your terminal:

streamlit run app.py

Your web browser will open with the application, ready for you to upload images.