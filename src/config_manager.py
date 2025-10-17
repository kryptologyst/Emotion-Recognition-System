"""
Configuration management for the emotion recognition system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for emotion detection models."""
    image_model_name: str = "microsoft/DialoGPT-medium"
    text_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    device: str = "auto"
    confidence_threshold: float = 0.5
    max_faces: int = 10


@dataclass
class UIConfig:
    """Configuration for user interface."""
    title: str = "Modern Emotion Recognition System"
    theme: str = "light"
    debug_mode: bool = False
    port: int = 8501


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'model': asdict(self.model),
            'ui': asdict(self.ui),
            'logging': asdict(self.logging)
        }


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/config.yaml")
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig object
        """
        if self._config is not None:
            return self._config
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        config_dict = yaml.safe_load(f)
                    else:
                        config_dict = json.load(f)
                
                self._config = AppConfig.from_dict(config_dict)
                logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self._config = self._get_default_config()
        else:
            logger.info("No config file found, using defaults")
            self._config = self._get_default_config()
        
        return self._config
    
    def save_config(self, config: AppConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            config: AppConfig object to save
        """
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = config.to_dict()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _get_default_config(self) -> AppConfig:
        """Get default configuration."""
        return AppConfig(
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def setup_logging(self, config: AppConfig) -> None:
        """
        Setup logging based on configuration.
        
        Args:
            config: AppConfig object
        """
        logging.basicConfig(
            level=getattr(logging, config.logging.level.upper()),
            format=config.logging.format,
            filename=config.logging.file_path
        )


def create_default_config_file(config_path: Path) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to create the config file
    """
    config_manager = ConfigManager(config_path)
    default_config = config_manager._get_default_config()
    config_manager.save_config(default_config)
    logger.info(f"Created default config file at {config_path}")


if __name__ == "__main__":
    # Example usage
    config_path = Path("config/config.yaml")
    config_manager = ConfigManager(config_path)
    
    # Load configuration
    config = config_manager.load_config()
    print(f"Loaded config: {config}")
    
    # Setup logging
    config_manager.setup_logging(config)
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        create_default_config_file(config_path)
