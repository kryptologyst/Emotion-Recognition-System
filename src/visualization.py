"""
Visualization utilities for emotion recognition results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from emotion_detector import EmotionResult


class EmotionVisualizer:
    """Visualization utilities for emotion recognition results."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = {
            'happy': '#FFD700',
            'sad': '#4169E1',
            'angry': '#FF4500',
            'fear': '#8B0000',
            'surprise': '#FF69B4',
            'disgust': '#32CD32',
            'neutral': '#808080',
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#FF4500'
        }
    
    def plot_emotion_distribution(
        self, 
        results: List[EmotionResult], 
        title: str = "Emotion Distribution",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot emotion distribution as a bar chart.
        
        Args:
            results: List of emotion detection results
            title: Chart title
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to visualize")
            return
        
        # Count emotions
        emotion_counts = {}
        for result in results:
            emotion = result.emotion.lower()
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Create plot
        emotions = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [self.colors.get(emotion, '#808080') for emotion in emotions]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(emotions, counts, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Emotions', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confidence_distribution(
        self, 
        results: List[EmotionResult], 
        title: str = "Confidence Distribution",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot confidence score distribution.
        
        Args:
            results: List of emotion detection results
            title: Chart title
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to visualize")
            return
        
        confidences = [result.confidence for result in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2f}')
        plt.axvline(np.median(confidences), color='green', linestyle='--', 
                   label=f'Median: {np.median(confidences):.2f}')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_emotion_timeline(
        self, 
        results_history: List[Dict[str, Any]], 
        title: str = "Emotion Timeline",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot emotion changes over time.
        
        Args:
            results_history: List of detection history entries
            title: Chart title
            save_path: Optional path to save the plot
        """
        if not results_history:
            print("No history to visualize")
            return
        
        # Extract timeline data
        timeline_data = []
        for i, entry in enumerate(results_history):
            for result in entry['results']:
                timeline_data.append({
                    'time': i,
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'type': entry['type']
                })
        
        if not timeline_data:
            print("No timeline data to visualize")
            return
        
        df = pd.DataFrame(timeline_data)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot emotion changes
        emotion_colors = [self.colors.get(emotion, '#808080') for emotion in df['emotion']]
        scatter = ax1.scatter(df['time'], df['emotion'], c=emotion_colors, 
                             s=df['confidence']*100, alpha=0.7)
        ax1.set_title('Emotion Changes Over Time', fontweight='bold')
        ax1.set_xlabel('Detection Number')
        ax1.set_ylabel('Emotion')
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence over time
        ax2.plot(df['time'], df['confidence'], marker='o', alpha=0.7)
        ax2.set_title('Confidence Over Time', fontweight='bold')
        ax2.set_xlabel('Detection Number')
        ax2.set_ylabel('Confidence Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(
        self, 
        results_history: List[Dict[str, Any]], 
        title: str = "Emotion Recognition Dashboard"
    ) -> None:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            results_history: List of detection history entries
            title: Dashboard title
        """
        if not results_history:
            print("No history to visualize")
            return
        
        # Prepare data
        all_results = []
        for entry in results_history:
            for result in entry['results']:
                all_results.append({
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'type': entry['type'],
                    'input': entry['input'][:50] + '...' if len(entry['input']) > 50 else entry['input']
                })
        
        df = pd.DataFrame(all_results)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Emotion Distribution', 'Confidence by Emotion', 
                          'Detection Types', 'Confidence Distribution'),
            specs=[[{"type": "pie"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Emotion distribution pie chart
        emotion_counts = df['emotion'].value_counts()
        fig.add_trace(
            go.Pie(labels=emotion_counts.index, values=emotion_counts.values,
                   name="Emotions"),
            row=1, col=1
        )
        
        # Confidence by emotion box plot
        for emotion in df['emotion'].unique():
            emotion_data = df[df['emotion'] == emotion]['confidence']
            fig.add_trace(
                go.Box(y=emotion_data, name=emotion, showlegend=False),
                row=1, col=2
            )
        
        # Detection types bar chart
        type_counts = df['type'].value_counts()
        fig.add_trace(
            go.Bar(x=type_counts.index, y=type_counts.values, name="Types"),
            row=2, col=1
        )
        
        # Confidence distribution histogram
        fig.add_trace(
            go.Histogram(x=df['confidence'], name="Confidence", nbinsx=20),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800
        )
        
        fig.show()
    
    def generate_report(
        self, 
        results_history: List[Dict[str, Any]], 
        output_path: Path
    ) -> None:
        """
        Generate a comprehensive visualization report.
        
        Args:
            results_history: List of detection history entries
            output_path: Path to save the report
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract all results
        all_results = []
        for entry in results_history:
            for result in entry['results']:
                all_results.append(EmotionResult(
                    emotion=result['emotion'],
                    confidence=result['confidence'],
                    face_detected=result.get('face_detected', False)
                ))
        
        if not all_results:
            print("No results to generate report for")
            return
        
        # Generate visualizations
        self.plot_emotion_distribution(
            all_results, 
            save_path=output_path / "emotion_distribution.png"
        )
        
        self.plot_confidence_distribution(
            all_results, 
            save_path=output_path / "confidence_distribution.png"
        )
        
        self.plot_emotion_timeline(
            results_history, 
            save_path=output_path / "emotion_timeline.png"
        )
        
        # Generate summary statistics
        stats = {
            'total_detections': len(all_results),
            'unique_emotions': len(set(r.emotion for r in all_results)),
            'average_confidence': np.mean([r.confidence for r in all_results]),
            'emotion_counts': {emotion: sum(1 for r in all_results if r.emotion == emotion) 
                             for emotion in set(r.emotion for r in all_results)},
            'face_detection_rate': sum(1 for r in all_results if r.face_detected) / len(all_results)
        }
        
        # Save statistics
        import json
        with open(output_path / "summary_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Report generated successfully in {output_path}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    # Create sample data
    sample_results = [
        EmotionResult("happy", 0.85, face_detected=True),
        EmotionResult("sad", 0.72, face_detected=True),
        EmotionResult("angry", 0.91, face_detected=True),
        EmotionResult("happy", 0.78, face_detected=True),
        EmotionResult("neutral", 0.65, face_detected=False),
    ]
    
    sample_history = [
        {
            'type': 'text',
            'input': 'I am so happy today!',
            'results': [{'emotion': 'happy', 'confidence': 0.85, 'face_detected': False}]
        },
        {
            'type': 'image',
            'input': 'uploaded_image.jpg',
            'results': [{'emotion': 'sad', 'confidence': 0.72, 'face_detected': True}]
        }
    ]
    
    # Create visualizer
    visualizer = EmotionVisualizer()
    
    # Generate visualizations
    visualizer.plot_emotion_distribution(sample_results)
    visualizer.plot_confidence_distribution(sample_results)
    visualizer.plot_emotion_timeline(sample_history)
    
    # Generate report
    report_path = Path("reports")
    visualizer.generate_report(sample_history, report_path)


if __name__ == "__main__":
    demo_visualization()
