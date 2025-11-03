"""
Configuration Management for GOP Analysis Experiments

Handles loading and validation of experiment configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentConfig:
    """
    Manages experiment configuration for GOP analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize configuration from file or dictionary.
        
        Args:
            config_path: Path to YAML config file
            config_dict: Configuration dictionary
        """
        if config_path:
            self.config = self._load_from_file(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        self._validate()
    
    def _load_from_file(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def _validate(self):
        """Validate configuration structure."""
        required_sections = ['experiment', 'gop_tracking', 'storage']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        logger.info("Configuration validated successfully")
    
    @property
    def experiment_name(self) -> str:
        return self.config['experiment'].get('name', 'unnamed_experiment')
    
    @property
    def gop_tracking(self) -> Dict:
        return self.config['gop_tracking']
    
    @property
    def storage(self) -> Dict:
        return self.config['storage']
    
    @property
    def training(self) -> Dict:
        return self.config.get('training', {})
    
    @property
    def model_config(self) -> Dict:
        return self.config.get('model', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def to_dict(self) -> Dict:
        """Get full configuration as dictionary."""
        return self.config.copy()

