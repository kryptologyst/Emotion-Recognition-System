"""
Command Line Interface for the Modern Emotion Recognition System.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional
import logging

# Import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))
from emotion_detector import ModernEmotionDetector, EmotionResult, create_synthetic_dataset
from config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_emotion_results(results: List[EmotionResult], title: str = "Detection Results") -> None:
    """Print emotion detection results to console."""
    if not results:
        print("❌ No emotions detected.")
        return
    
    print(f"\n📊 {title}")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Emotion: {result.emotion.title()}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Face Detected: {'✅' if result.face_detected else '❌'}")
        
        if result.bounding_box:
            x, y, w, h = result.bounding_box
            print(f"   Bounding Box: ({x}, {y}, {w}, {h})")
        
        # Simple confidence bar
        bar_length = 20
        filled_length = int(bar_length * result.confidence)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"   Confidence: [{bar}] {result.confidence:.1%}")


def analyze_text_cli(text: str, detector: ModernEmotionDetector) -> None:
    """Analyze emotions in text via CLI."""
    print(f"\n📝 Analyzing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    results = detector.detect_emotions_from_text(text)
    print_emotion_results(results, "Text Emotion Analysis")


def analyze_image_cli(image_path: Path, detector: ModernEmotionDetector) -> None:
    """Analyze emotions in image via CLI."""
    if not image_path.exists():
        print(f"❌ Error: Image file '{image_path}' not found.")
        return
    
    print(f"\n🖼️ Analyzing image: {image_path}")
    
    results = detector.detect_emotions_from_image(image_path)
    print_emotion_results(results, "Image Emotion Analysis")


def webcam_detection_cli(detector: ModernEmotionDetector) -> None:
    """Real-time emotion detection from webcam via CLI."""
    print("\n📹 Starting webcam emotion detection...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            result = detector.detect_emotions_from_webcam()
            
            if result:
                print(f"\r😊 Detected: {result.emotion.title()} ({result.confidence:.2f})", end="", flush=True)
            else:
                print(f"\r❌ No face detected", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n👋 Webcam detection stopped.")


def batch_analysis_cli(input_file: Path, detector: ModernEmotionDetector) -> None:
    """Batch analysis of multiple inputs via CLI."""
    if not input_file.exists():
        print(f"❌ Error: Input file '{input_file}' not found.")
        return
    
    print(f"\n📁 Batch analyzing file: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Error: Input file must contain a JSON array.")
            return
        
        all_results = []
        
        for i, item in enumerate(data):
            print(f"\n--- Processing item {i+1}/{len(data)} ---")
            
            if 'text' in item:
                results = detector.detect_emotions_from_text(item['text'])
                all_results.append({
                    'type': 'text',
                    'input': item['text'],
                    'results': [
                        {
                            'emotion': r.emotion,
                            'confidence': r.confidence,
                            'face_detected': r.face_detected
                        } for r in results
                    ]
                })
                print_emotion_results(results, f"Text Analysis {i+1}")
            
            elif 'image_path' in item:
                image_path = Path(item['image_path'])
                results = detector.detect_emotions_from_image(image_path)
                all_results.append({
                    'type': 'image',
                    'input': str(image_path),
                    'results': [
                        {
                            'emotion': r.emotion,
                            'confidence': r.confidence,
                            'face_detected': r.face_detected,
                            'bounding_box': r.bounding_box
                        } for r in results
                    ]
                })
                print_emotion_results(results, f"Image Analysis {i+1}")
        
        # Save batch results
        output_file = input_file.parent / f"{input_file.stem}_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✅ Batch analysis complete! Results saved to: {output_file}")
    
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in input file.")
    except Exception as e:
        print(f"❌ Error during batch analysis: {e}")


def create_sample_data(output_file: Path) -> None:
    """Create sample data file for batch analysis."""
    sample_data = [
        {
            "text": "I'm so excited about this new project!",
            "type": "text"
        },
        {
            "text": "This is absolutely terrible and disappointing.",
            "type": "text"
        },
        {
            "text": "I can't believe this happened!",
            "type": "text"
        },
        {
            "image_path": "data/sample_image.jpg",
            "type": "image"
        }
    ]
    
    with open(output_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✅ Sample data file created: {output_file}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Modern Emotion Recognition System - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze text
  python cli.py --text "I'm so happy today!"
  
  # Analyze image
  python cli.py --image path/to/image.jpg
  
  # Webcam detection
  python cli.py --webcam
  
  # Batch analysis
  python cli.py --batch input.json
  
  # Create sample data
  python cli.py --create-sample sample.json
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Text to analyze for emotions')
    input_group.add_argument('--image', type=Path, help='Path to image file to analyze')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    input_group.add_argument('--batch', type=Path, help='JSON file with batch inputs to analyze')
    input_group.add_argument('--create-sample', type=Path, help='Create sample data file')
    
    # Optional arguments
    parser.add_argument('--output', type=Path, help='Output file for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--config', type=Path, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    # Initialize detector
    print("🚀 Initializing Modern Emotion Recognition System...")
    detector = ModernEmotionDetector()
    
    # Handle different input types
    try:
        if args.text:
            analyze_text_cli(args.text, detector)
        
        elif args.image:
            analyze_image_cli(args.image, detector)
        
        elif args.webcam:
            webcam_detection_cli(detector)
        
        elif args.batch:
            batch_analysis_cli(args.batch, detector)
        
        elif args.create_sample:
            create_sample_data(args.create_sample)
        
        print("\n✅ Analysis complete!")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
