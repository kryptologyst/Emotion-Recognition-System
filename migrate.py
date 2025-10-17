"""
Migration script to help transition from the old FER library to the modern system.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))
from emotion_detector import ModernEmotionDetector, EmotionResult


def migrate_fer_code(input_file: Path, output_file: Path) -> None:
    """
    Migrate FER-based code to the new system.
    
    Args:
        input_file: Path to file containing old FER code
        output_file: Path to save migrated code
    """
    if not input_file.exists():
        print(f"❌ Input file {input_file} not found")
        return
    
    # Read the old code
    with open(input_file, 'r') as f:
        old_code = f.read()
    
    # Simple migration patterns
    migrations = [
        # Import statements
        ("from fer import FER", "from src.emotion_detector import ModernEmotionDetector"),
        ("import fer", "from src.emotion_detector import ModernEmotionDetector"),
        
        # Detector initialization
        ("detector = FER(mtcnn=True)", "detector = ModernEmotionDetector()"),
        ("detector = FER()", "detector = ModernEmotionDetector()"),
        
        # Method calls
        ("detector.detect_emotions(", "detector.detect_emotions_from_image("),
        
        # Result handling (this is more complex and would need custom logic)
        ("result[0][\"emotions\"]", "# Migrated: results = detector.detect_emotions_from_image(image_path)\n    # for result in results:\n    #     print(f\"Emotion: {result.emotion}, Confidence: {result.confidence}\")"),
    ]
    
    # Apply migrations
    migrated_code = old_code
    for old_pattern, new_pattern in migrations:
        migrated_code = migrated_code.replace(old_pattern, new_pattern)
    
    # Add migration comments
    migrated_code = f"""# Migrated from FER library to Modern Emotion Recognition System
# Original file: {input_file}
# Migration date: {Path(__file__).stat().st_mtime}

{migrated_code}

# Migration Notes:
# - FER.detect_emotions() -> ModernEmotionDetector.detect_emotions_from_image()
# - Results format changed from dict to EmotionResult objects
# - Added support for text emotion detection
# - Added configuration management
# - Added comprehensive testing
"""
    
    # Save migrated code
    with open(output_file, 'w') as f:
        f.write(migrated_code)
    
    print(f"✅ Code migrated from {input_file} to {output_file}")


def create_migration_guide(output_file: Path) -> None:
    """Create a comprehensive migration guide."""
    guide_content = """# Migration Guide: FER Library to Modern Emotion Recognition System

## Overview
This guide helps you migrate from the deprecated FER library to our modern emotion recognition system.

## Key Changes

### 1. Import Changes
```python
# Old (FER)
from fer import FER

# New (Modern System)
from src.emotion_detector import ModernEmotionDetector
```

### 2. Detector Initialization
```python
# Old (FER)
detector = FER(mtcnn=True)

# New (Modern System)
detector = ModernEmotionDetector()
```

### 3. Emotion Detection
```python
# Old (FER)
result = detector.detect_emotions(image)
emotions = result[0]["emotions"]
top_emotion = max(emotions, key=emotions.get)

# New (Modern System)
results = detector.detect_emotions_from_image(image_path)
for result in results:
    print(f"Emotion: {result.emotion}, Confidence: {result.confidence}")
```

### 4. Result Format
```python
# Old (FER) - Dictionary format
{
    "emotions": {
        "angry": 0.1,
        "disgust": 0.05,
        "fear": 0.02,
        "happy": 0.8,
        "sad": 0.01,
        "surprise": 0.01,
        "neutral": 0.01
    },
    "bounding_box": [x, y, w, h]
}

# New (Modern System) - EmotionResult objects
EmotionResult(
    emotion="happy",
    confidence=0.8,
    bounding_box=(x, y, w, h),
    face_detected=True
)
```

## New Features

### 1. Text Emotion Detection
```python
# Analyze emotions in text
results = detector.detect_emotions_from_text("I'm so happy today!")
```

### 2. Configuration Management
```python
from src.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config()
```

### 3. Web Interface
```bash
streamlit run web_app/app.py
```

### 4. Command Line Interface
```bash
python src/cli.py --text "Hello world!"
python src/cli.py --image path/to/image.jpg
```

### 5. Batch Processing
```python
# Process multiple inputs
batch_results = []
for image_path in image_paths:
    results = detector.detect_emotions_from_image(image_path)
    batch_results.extend(results)
```

## Migration Steps

1. **Install the new system**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update imports**:
   Replace FER imports with ModernEmotionDetector imports

3. **Update detector initialization**:
   Replace FER() with ModernEmotionDetector()

4. **Update emotion detection calls**:
   Replace detect_emotions() with detect_emotions_from_image()

5. **Update result handling**:
   Adapt your code to work with EmotionResult objects

6. **Test the migration**:
   ```bash
   python -m pytest tests/ -v
   ```

## Common Issues and Solutions

### Issue: "No module named 'fer'"
**Solution**: The FER library is deprecated. Use the new ModernEmotionDetector instead.

### Issue: Different result format
**Solution**: Update your code to work with EmotionResult objects instead of dictionaries.

### Issue: Performance differences
**Solution**: The new system uses modern transformers and may have different performance characteristics.

## Support
- Check the README.md for detailed documentation
- Run tests to verify your migration
- Use the CLI for quick testing
- Check GitHub issues for common problems

## Benefits of Migration
- ✅ Modern AI models (Hugging Face transformers)
- ✅ Better accuracy and performance
- ✅ Text emotion detection
- ✅ Web interface
- ✅ Comprehensive testing
- ✅ Configuration management
- ✅ Better documentation
- ✅ Active maintenance and support
"""
    
    with open(output_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Migration guide created: {output_file}")


def analyze_fer_usage(directory: Path) -> Dict[str, Any]:
    """Analyze FER library usage in a directory."""
    fer_files = []
    fer_imports = []
    
    for py_file in directory.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            if "fer" in content.lower() or "FER" in content:
                fer_files.append(str(py_file))
                
                # Find specific FER usage patterns
                if "from fer import" in content:
                    fer_imports.append(str(py_file))
                
        except Exception as e:
            print(f"⚠️  Could not analyze {py_file}: {e}")
    
    return {
        "fer_files": fer_files,
        "fer_imports": fer_imports,
        "total_files": len(fer_files),
        "files_with_imports": len(fer_imports)
    }


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate from FER library to Modern Emotion Recognition System")
    parser.add_argument("--analyze", type=Path, help="Analyze directory for FER usage")
    parser.add_argument("--migrate-file", type=Path, help="Migrate specific Python file")
    parser.add_argument("--output", type=Path, help="Output file for migration")
    parser.add_argument("--create-guide", action="store_true", help="Create migration guide")
    
    args = parser.parse_args()
    
    if args.analyze:
        print(f"🔍 Analyzing directory: {args.analyze}")
        analysis = analyze_fer_usage(args.analyze)
        
        print(f"\n📊 Analysis Results:")
        print(f"- Files with FER usage: {analysis['total_files']}")
        print(f"- Files with FER imports: {analysis['files_with_imports']}")
        
        if analysis['fer_files']:
            print(f"\n📁 Files to migrate:")
            for file_path in analysis['fer_files']:
                print(f"  - {file_path}")
    
    elif args.migrate_file:
        output_file = args.output or args.migrate_file.parent / f"{args.migrate_file.stem}_migrated.py"
        migrate_fer_code(args.migrate_file, output_file)
    
    elif args.create_guide:
        output_file = args.output or Path("MIGRATION_GUIDE.md")
        create_migration_guide(output_file)
    
    else:
        print("❌ Please specify an action: --analyze, --migrate-file, or --create-guide")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
