import streamlit as st
from ultralytics import YOLO
from PIL import Image
import glob
import os
from pathlib import Path

def find_latest_model_path():
    """
    Finds the path to the 'best.pt' file from the latest training run,
    looking in the project's root directory.
    """
    try:
        # Search for the 'runs' directory in the current working directory.
        # This is more robust for deployment environments like Streamlit Cloud.
        list_of_dirs = glob.glob('runs/detect/yolov8s_carsdd_fine_tuned*')
        if not list_of_dirs:
            return None
        # Get the latest directory based on name/creation time
        latest_dir = max(list_of_dirs, key=os.path.getctime)
        model_path = Path(latest_dir) / 'weights' / 'best.pt'
        return model_path
    except Exception:
        return None

def main():
    """
    Main function to run the Streamlit application.
    Allows users to upload an image and view YOLOv8 inference results.
    """
    st.set_page_config(
        page_title="Car Damage Detection with YOLOv8",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("🚗 Car Damage Detection with YOLOv8")
    st.write("Upload an image of a car to detect potential damages.")

    # --- Model Loading ---
    model_path = find_latest_model_path()

    if model_path is None or not model_path.exists():
        st.error("Error: Could not find a trained model ('best.pt').")
        st.info("Please ensure a 'runs' folder with a trained model exists in your GitHub repository.")
        return

    try:
        # Load the fine-tuned YOLOv8 model
        st.success(f"Successfully loaded model: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # --- Image Upload ---
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown("---")
            st.subheader("Detection Results")

            # --- Run Inference ---
            results = model([image])

            # Process and display results
            for result in results:
                result_image_pil = Image.fromarray(result.plot()[:, :, ::-1])
                st.image(
                    result_image_pil,
                    caption="Detection Result",
                    use_column_width=True,
                )

                if result.boxes:
                    st.write("Detected Objects:")
                    for box in result.boxes:
                        class_id = int(box.cls)
                        class_name = model.names[class_id]
                        confidence = float(box.conf)
                        st.info(f"- **{class_name.title()}** (Confidence: {confidence:.2f})")
                else:
                    st.success("No damages or cars detected in the image.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()