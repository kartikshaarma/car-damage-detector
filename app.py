import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path
import glob
import os

def find_latest_model_path():
    """Finds the path to the 'best.pt' file from the latest training run."""
    try:
        # The 'runs' folder is often created in the user's home directory
        # by default. We'll construct a path to search there.
        home_dir = Path.home()
        search_path = home_dir / 'runs' / 'detect' / 'yolov8s_carsdd_fine_tuned*'
        
        # Use glob to find all matching directories
        list_of_dirs = glob.glob(str(search_path))
        
        if not list_of_dirs:
            # As a fallback, check the local directory as well
            list_of_dirs = glob.glob('runs/detect/yolov8s_carsdd_fine_tuned*')
            if not list_of_dirs:
                return None

        # Get the latest directory based on creation time
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
    # Automatically find the path to the latest trained model.
    model_path = find_latest_model_path()

    if model_path is None or not model_path.exists():
        st.error("Error: Could not find a trained model ('best.pt').")
        st.info("Please ensure you have run the training script (`train.py`) successfully and check if a 'runs' folder exists.")
        return

    try:
        # Load the fine-tuned YOLOv8 model
        st.success(f"Successfully loaded model from: {model_path}")
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
            # Open the uploaded image
            image = Image.open(uploaded_file)

            # Display the original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown("---")
            st.subheader("Detection Results")

            # --- Run Inference ---
            # The model expects a list of images
            results = model([image])

            # Process and display results
            for result in results:
                # Convert the result image with bounding boxes to a displayable format
                result_image_pil = Image.fromarray(result.plot()[:, :, ::-1])
                st.image(
                    result_image_pil,
                    caption="Detection Result",
                    use_column_width=True,
                )

                # Display detected objects and their confidence scores
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

