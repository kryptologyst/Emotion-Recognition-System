"""
Streamlit web interface for the Modern Emotion Recognition System.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import json
from typing import List, Dict, Any

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent))
from emotion_detector import ModernEmotionDetector, EmotionResult, create_synthetic_dataset
from config_manager import ConfigManager, AppConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Modern Emotion Recognition System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-bar {
        background-color: #1f77b4;
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stat-item {
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []


def load_configuration() -> AppConfig:
    """Load application configuration."""
    config_manager = ConfigManager()
    return config_manager.load_config()


def initialize_detector() -> ModernEmotionDetector:
    """Initialize the emotion detector."""
    if st.session_state.detector is None:
        with st.spinner("Loading emotion detection models..."):
            st.session_state.detector = ModernEmotionDetector()
    return st.session_state.detector


def display_emotion_results(results: List[EmotionResult], title: str = "Detection Results") -> None:
    """Display emotion detection results."""
    if not results:
        st.warning("No emotions detected.")
        return
    
    st.subheader(title)
    
    for i, result in enumerate(results):
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Emotion:** {result.emotion.title()}")
            
            with col2:
                st.markdown(f"**Confidence:** {result.confidence:.2f}")
            
            with col3:
                if result.face_detected:
                    st.markdown("✅ Face Detected")
                else:
                    st.markdown("❌ No Face")
            
            # Confidence bar
            confidence_percent = result.confidence * 100
            st.markdown(f"""
            <div class="confidence-bar" style="width: {confidence_percent}%;"></div>
            """, unsafe_allow_html=True)
            
            if result.bounding_box:
                st.markdown(f"**Bounding Box:** {result.bounding_box}")


def analyze_text_emotions(text: str) -> None:
    """Analyze emotions in text."""
    detector = initialize_detector()
    
    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return
    
    results = detector.detect_emotions_from_text(text)
    
    if results:
        display_emotion_results(results, "Text Emotion Analysis")
        
        # Add to history
        st.session_state.detection_history.append({
            'type': 'text',
            'input': text,
            'results': [
                {
                    'emotion': r.emotion,
                    'confidence': r.confidence,
                    'face_detected': r.face_detected
                } for r in results
            ]
        })
    else:
        st.error("Failed to analyze text emotions.")


def analyze_image_emotions(image: Image.Image) -> None:
    """Analyze emotions in an image."""
    detector = initialize_detector()
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Save temporary image
    temp_path = Path("temp_image.jpg")
    cv2.imwrite(str(temp_path), image_cv)
    
    try:
        results = detector.detect_emotions_from_image(temp_path)
        
        if results:
            display_emotion_results(results, "Image Emotion Analysis")
            
            # Display image with bounding boxes
            if any(r.bounding_box for r in results):
                display_image = image_cv.copy()
                for result in results:
                    if result.bounding_box:
                        x, y, w, h = result.bounding_box
                        cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_image, f"{result.emotion}: {result.confidence:.2f}", 
                                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), 
                        caption="Image with Emotion Detection", use_column_width=True)
            
            # Add to history
            st.session_state.detection_history.append({
                'type': 'image',
                'input': 'Uploaded Image',
                'results': [
                    {
                        'emotion': r.emotion,
                        'confidence': r.confidence,
                        'face_detected': r.face_detected,
                        'bounding_box': r.bounding_box
                    } for r in results
                ]
            })
        else:
            st.warning("No emotions detected in the image.")
    
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


def webcam_emotion_detection() -> None:
    """Real-time emotion detection from webcam."""
    detector = initialize_detector()
    
    st.subheader("Real-time Emotion Detection")
    st.write("Click the button below to capture and analyze emotions from your webcam.")
    
    if st.button("Capture from Webcam", key="webcam_capture"):
        with st.spinner("Capturing from webcam..."):
            result = detector.detect_emotions_from_webcam()
            
            if result:
                display_emotion_results([result], "Webcam Emotion Detection")
                
                # Add to history
                st.session_state.detection_history.append({
                    'type': 'webcam',
                    'input': 'Webcam Capture',
                    'results': [{
                        'emotion': result.emotion,
                        'confidence': result.confidence,
                        'face_detected': result.face_detected,
                        'bounding_box': result.bounding_box
                    }]
                })
            else:
                st.warning("No face detected in webcam feed.")


def display_statistics() -> None:
    """Display emotion detection statistics."""
    if not st.session_state.detection_history:
        st.info("No detection history available yet.")
        return
    
    st.subheader("Detection Statistics")
    
    # Calculate overall statistics
    all_emotions = []
    for entry in st.session_state.detection_history:
        for result in entry['results']:
            all_emotions.append(result['emotion'])
    
    if all_emotions:
        emotion_counts = {}
        for emotion in all_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Display emotion distribution
        st.write("**Emotion Distribution:**")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_emotions)) * 100
            st.write(f"{emotion.title()}: {count} ({percentage:.1f}%)")
        
        # Display recent detections
        st.write("**Recent Detections:**")
        for i, entry in enumerate(st.session_state.detection_history[-5:]):
            st.write(f"{i+1}. {entry['type'].title()}: {entry['input'][:50]}...")
            for result in entry['results']:
                st.write(f"   - {result['emotion'].title()} ({result['confidence']:.2f})")


def main():
    """Main application function."""
    # Load configuration
    if st.session_state.config is None:
        st.session_state.config = load_configuration()
    
    # Header
    st.markdown('<h1 class="main-header">😊 Modern Emotion Recognition System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Text Analysis", "Image Analysis", "Webcam Detection", "Statistics", "Settings"]
    )
    
    # Main content based on selected page
    if page == "Text Analysis":
        st.header("📝 Text Emotion Analysis")
        st.write("Analyze emotions in text using state-of-the-art NLP models.")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Analyze Text Emotions"):
            analyze_text_emotions(text_input)
    
    elif page == "Image Analysis":
        st.header("🖼️ Image Emotion Analysis")
        st.write("Upload an image to detect emotions in faces.")
        
        uploaded_file = st.file_uploader(
            "Choose an image file:",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image Emotions"):
                analyze_image_emotions(image)
    
    elif page == "Webcam Detection":
        st.header("📹 Real-time Emotion Detection")
        webcam_emotion_detection()
    
    elif page == "Statistics":
        st.header("📊 Detection Statistics")
        display_statistics()
        
        # Export history
        if st.session_state.detection_history:
            if st.button("Export Detection History"):
                history_json = json.dumps(st.session_state.detection_history, indent=2)
                st.download_button(
                    label="Download History as JSON",
                    data=history_json,
                    file_name="emotion_detection_history.json",
                    mime="application/json"
                )
    
    elif page == "Settings":
        st.header("⚙️ Settings")
        
        st.subheader("Model Configuration")
        st.write("Current model settings:")
        
        config = st.session_state.config
        st.write(f"- Text Model: {config.model.text_model_name}")
        st.write(f"- Image Model: {config.model.image_model_name}")
        st.write(f"- Device: {config.model.device}")
        st.write(f"- Confidence Threshold: {config.model.confidence_threshold}")
        
        st.subheader("Data Management")
        if st.button("Create Synthetic Dataset"):
            with st.spinner("Creating synthetic dataset..."):
                data_dir = Path("data")
                create_synthetic_dataset(data_dir, 100)
                st.success("Synthetic dataset created successfully!")
        
        if st.button("Clear Detection History"):
            st.session_state.detection_history = []
            st.success("Detection history cleared!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Hugging Face Transformers, and OpenCV | "
        "Modern Emotion Recognition System v2.0"
    )


if __name__ == "__main__":
    main()
