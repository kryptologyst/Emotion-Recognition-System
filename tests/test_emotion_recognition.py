"""
Test suite for the Modern Emotion Recognition System.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import cv2
from PIL import Image

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from emotion_detector import ModernEmotionDetector, EmotionResult, create_synthetic_dataset
from config_manager import ConfigManager, AppConfig, ModelConfig, UIConfig, LoggingConfig


class TestEmotionDetector(unittest.TestCase):
    """Test cases for the ModernEmotionDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ModernEmotionDetector()
    
    def test_init(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.emotion_labels)
        self.assertIsNotNone(self.detector.face_cascade)
    
    def test_detect_emotions_from_text(self):
        """Test text emotion detection."""
        test_text = "I'm so happy and excited!"
        results = self.detector.detect_emotions_from_text(test_text)
        
        # Should return a list
        self.assertIsInstance(results, list)
        
        # If text pipeline is available, should have results
        if self.detector.text_pipeline:
            self.assertGreater(len(results), 0)
            for result in results:
                self.assertIsInstance(result, EmotionResult)
                self.assertIsInstance(result.emotion, str)
                self.assertIsInstance(result.confidence, float)
                self.assertGreaterEqual(result.confidence, 0.0)
                self.assertLessEqual(result.confidence, 1.0)
    
    def test_detect_emotions_from_text_empty(self):
        """Test text emotion detection with empty input."""
        results = self.detector.detect_emotions_from_text("")
        self.assertEqual(len(results), 0)
    
    def test_detect_faces(self):
        """Test face detection functionality."""
        # Create a simple test image with a face-like pattern
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw a simple face-like rectangle
        cv2.rectangle(test_image, (20, 20), (80, 80), (255, 255, 255), -1)
        
        faces = self.detector.detect_faces(test_image)
        self.assertIsInstance(faces, list)
    
    def test_detect_emotions_from_image_nonexistent(self):
        """Test image emotion detection with non-existent file."""
        results = self.detector.detect_emotions_from_image("nonexistent.jpg")
        self.assertEqual(len(results), 0)
    
    def test_get_emotion_statistics(self):
        """Test emotion statistics calculation."""
        # Create mock results
        results = [
            EmotionResult("happy", 0.8, face_detected=True),
            EmotionResult("sad", 0.6, face_detected=True),
            EmotionResult("happy", 0.9, face_detected=True)
        ]
        
        stats = self.detector.get_emotion_statistics(results)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("happy", stats)
        self.assertIn("sad", stats)
        self.assertEqual(stats["happy"], 66.66666666666666)  # 2/3 * 100
        self.assertEqual(stats["sad"], 33.33333333333333)   # 1/3 * 100
    
    def test_get_emotion_statistics_empty(self):
        """Test emotion statistics with empty results."""
        stats = self.detector.get_emotion_statistics([])
        self.assertEqual(stats, {})


class TestConfigManager(unittest.TestCase):
    """Test cases for the ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config_manager = ConfigManager(self.config_path)
        config = config_manager._get_default_config()
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.ui, UIConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager(self.config_path)
        original_config = config_manager._get_default_config()
        
        # Save config
        config_manager.save_config(original_config)
        self.assertTrue(self.config_path.exists())
        
        # Load config
        loaded_config = config_manager.load_config()
        
        self.assertEqual(original_config.model.image_model_name, loaded_config.model.image_model_name)
        self.assertEqual(original_config.model.text_model_name, loaded_config.model.text_model_name)
        self.assertEqual(original_config.ui.title, loaded_config.ui.title)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'model': {
                'image_model_name': 'test_model',
                'text_model_name': 'test_text_model',
                'device': 'cpu',
                'confidence_threshold': 0.7,
                'max_faces': 5
            },
            'ui': {
                'title': 'Test App',
                'theme': 'dark',
                'debug_mode': True,
                'port': 8080
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'test format',
                'file_path': 'test.log'
            }
        }
        
        config = AppConfig.from_dict(config_dict)
        
        self.assertEqual(config.model.image_model_name, 'test_model')
        self.assertEqual(config.model.text_model_name, 'test_text_model')
        self.assertEqual(config.ui.title, 'Test App')
        self.assertEqual(config.logging.level, 'DEBUG')


class TestSyntheticDataset(unittest.TestCase):
    """Test cases for synthetic dataset creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        create_synthetic_dataset(self.data_dir, 10)
        
        # Check if directory was created
        self.assertTrue(self.data_dir.exists())
        
        # Check if data file was created
        data_file = self.data_dir / "synthetic_text_data.json"
        self.assertTrue(data_file.exists())
        
        # Check data content
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 10)
        
        for item in data:
            self.assertIn('text', item)
            self.assertIn('emotion', item)
            self.assertIn('sample_id', item)
            self.assertIsInstance(item['text'], str)
            self.assertIsInstance(item['emotion'], str)
            self.assertIsInstance(item['sample_id'], int)


class TestEmotionResult(unittest.TestCase):
    """Test cases for the EmotionResult dataclass."""
    
    def test_emotion_result_creation(self):
        """Test EmotionResult creation."""
        result = EmotionResult(
            emotion="happy",
            confidence=0.85,
            bounding_box=(10, 20, 30, 40),
            face_detected=True
        )
        
        self.assertEqual(result.emotion, "happy")
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.bounding_box, (10, 20, 30, 40))
        self.assertTrue(result.face_detected)
    
    def test_emotion_result_minimal(self):
        """Test EmotionResult with minimal parameters."""
        result = EmotionResult(emotion="sad", confidence=0.6)
        
        self.assertEqual(result.emotion, "sad")
        self.assertEqual(result.confidence, 0.6)
        self.assertIsNone(result.bounding_box)
        self.assertFalse(result.face_detected)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_text_analysis(self):
        """Test end-to-end text analysis workflow."""
        detector = ModernEmotionDetector()
        
        test_texts = [
            "I'm so happy today!",
            "This is terrible and disappointing.",
            "I can't believe this happened!"
        ]
        
        for text in test_texts:
            results = detector.detect_emotions_from_text(text)
            self.assertIsInstance(results, list)
            
            if results:  # If text pipeline is available
                for result in results:
                    self.assertIsInstance(result, EmotionResult)
                    self.assertIsInstance(result.emotion, str)
                    self.assertIsInstance(result.confidence, float)
    
    def test_config_and_detector_integration(self):
        """Test integration between config manager and detector."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()
            
            # Create detector with config
            detector = ModernEmotionDetector(
                device=config.model.device
            )
            
            self.assertIsNotNone(detector)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEmotionDetector,
        TestConfigManager,
        TestSyntheticDataset,
        TestEmotionResult,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
