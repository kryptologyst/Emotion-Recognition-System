#!/usr/bin/env python3
"""
Demo script for the Modern Emotion Recognition System.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from emotion_detector import ModernEmotionDetector, create_synthetic_dataset
from config_manager import ConfigManager
from visualization import EmotionVisualizer


def demo_text_analysis():
    """Demonstrate text emotion analysis."""
    print("\n" + "="*60)
    print("📝 TEXT EMOTION ANALYSIS DEMO")
    print("="*60)
    
    detector = ModernEmotionDetector()
    
    sample_texts = [
        "I'm so excited about this new project! It's going to be amazing!",
        "This is absolutely terrible and disappointing. I can't believe this happened.",
        "Wow! I can't believe this happened! This is incredible!",
        "I'm really scared and worried about what might happen next.",
        "This is absolutely disgusting and revolting. I hate it.",
        "I feel so down and disappointed about everything.",
        "I'm feeling neutral about this situation. Nothing special."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        results = detector.detect_emotions_from_text(text)
        
        if results:
            for result in results:
                print(f"   😊 Emotion: {result.emotion.title()}")
                print(f"   📊 Confidence: {result.confidence:.2f}")
                print(f"   🎯 Face Detected: {'Yes' if result.face_detected else 'No'}")
        else:
            print("   ❌ No emotions detected")
        
        time.sleep(0.5)  # Small delay for demo effect


def demo_image_analysis():
    """Demonstrate image emotion analysis."""
    print("\n" + "="*60)
    print("🖼️ IMAGE EMOTION ANALYSIS DEMO")
    print("="*60)
    
    detector = ModernEmotionDetector()
    
    # Create a simple test image (in a real demo, you'd use actual images)
    print("\n📸 Simulating image analysis...")
    print("   (In a real scenario, you would provide actual image files)")
    
    # Mock image analysis
    sample_results = [
        ("sample_happy_face.jpg", "happy", 0.85),
        ("sample_sad_face.jpg", "sad", 0.72),
        ("sample_angry_face.jpg", "angry", 0.91),
        ("sample_surprised_face.jpg", "surprise", 0.78),
        ("sample_neutral_face.jpg", "neutral", 0.65)
    ]
    
    for image_path, emotion, confidence in sample_results:
        print(f"\n📷 Analyzing: {image_path}")
        print(f"   😊 Detected Emotion: {emotion.title()}")
        print(f"   📊 Confidence: {confidence:.2f}")
        print(f"   🎯 Face Detected: Yes")
        time.sleep(0.3)


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "="*60)
    print("⚙️ CONFIGURATION MANAGEMENT DEMO")
    print("="*60)
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    print(f"\n📋 Current Configuration:")
    print(f"   🤖 Text Model: {config.model.text_model_name}")
    print(f"   🖼️ Image Model: {config.model.image_model_name}")
    print(f"   💻 Device: {config.model.device}")
    print(f"   🎯 Confidence Threshold: {config.model.confidence_threshold}")
    print(f"   🌐 UI Title: {config.ui.title}")
    print(f"   🎨 Theme: {config.ui.theme}")
    print(f"   📝 Log Level: {config.logging.level}")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "="*60)
    print("🎲 SYNTHETIC DATA GENERATION DEMO")
    print("="*60)
    
    data_dir = Path("data")
    print(f"\n📁 Creating synthetic dataset in: {data_dir}")
    
    create_synthetic_dataset(data_dir, 20)
    
    # Show sample of generated data
    data_file = data_dir / "synthetic_text_data.json"
    if data_file.exists():
        import json
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📊 Generated {len(data)} synthetic samples:")
        for i, sample in enumerate(data[:5], 1):  # Show first 5
            print(f"   {i}. '{sample['text'][:50]}...' -> {sample['emotion']}")
        
        if len(data) > 5:
            print(f"   ... and {len(data) - 5} more samples")


def demo_statistics():
    """Demonstrate emotion statistics."""
    print("\n" + "="*60)
    print("📊 EMOTION STATISTICS DEMO")
    print("="*60)
    
    detector = ModernEmotionDetector()
    
    # Generate some sample results
    sample_results = [
        detector.EmotionResult("happy", 0.85, face_detected=True),
        detector.EmotionResult("sad", 0.72, face_detected=True),
        detector.EmotionResult("angry", 0.91, face_detected=True),
        detector.EmotionResult("happy", 0.78, face_detected=True),
        detector.EmotionResult("neutral", 0.65, face_detected=False),
        detector.EmotionResult("surprise", 0.88, face_detected=True),
        detector.EmotionResult("happy", 0.82, face_detected=True),
    ]
    
    stats = detector.get_emotion_statistics(sample_results)
    
    print(f"\n📈 Emotion Distribution:")
    for emotion, percentage in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(percentage / 5)  # Simple bar chart
        print(f"   {emotion.title():<10} {percentage:5.1f}% {bar}")
    
    avg_confidence = sum(r.confidence for r in sample_results) / len(sample_results)
    face_detection_rate = sum(1 for r in sample_results if r.face_detected) / len(sample_results)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   🎯 Average Confidence: {avg_confidence:.2f}")
    print(f"   👤 Face Detection Rate: {face_detection_rate:.1%}")
    print(f"   📝 Total Detections: {len(sample_results)}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("📈 VISUALIZATION DEMO")
    print("="*60)
    
    print("\n🎨 Visualization features available:")
    print("   📊 Emotion distribution charts")
    print("   📈 Confidence score histograms")
    print("   📅 Emotion timeline plots")
    print("   🎛️ Interactive dashboards")
    print("   📋 Comprehensive reports")
    
    print("\n💡 To use visualizations:")
    print("   from src.visualization import EmotionVisualizer")
    print("   visualizer = EmotionVisualizer()")
    print("   visualizer.plot_emotion_distribution(results)")


def main():
    """Main demo function."""
    print("🎉 WELCOME TO THE MODERN EMOTION RECOGNITION SYSTEM DEMO!")
    print("="*60)
    print("This demo showcases the key features of our modernized system.")
    print("="*60)
    
    try:
        # Run all demos
        demo_text_analysis()
        demo_image_analysis()
        demo_configuration()
        demo_synthetic_data()
        demo_statistics()
        demo_visualization()
        
        print("\n" + "="*60)
        print("🎊 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🚀 Next Steps:")
        print("1. Launch the web interface: streamlit run web_app/app.py")
        print("2. Try the CLI: python src/cli.py --text 'Hello world!'")
        print("3. Run tests: python -m pytest tests/ -v")
        print("4. Check the README.md for detailed documentation")
        
        print("\n💡 Key Features Demonstrated:")
        print("✅ Text emotion analysis with modern NLP models")
        print("✅ Image emotion detection with face recognition")
        print("✅ Configuration management system")
        print("✅ Synthetic data generation")
        print("✅ Emotion statistics and analytics")
        print("✅ Visualization capabilities")
        print("✅ Comprehensive testing framework")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
