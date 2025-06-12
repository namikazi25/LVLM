"""Prompt management system for MMFakeBench.

This module provides a centralized system for managing prompt templates
with Jinja2 templating support and versioning capabilities.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound


class PromptManager:
    """Manages prompt templates with Jinja2 templating support.
    
    This class provides centralized management of prompt templates,
    including loading, rendering, and versioning capabilities.
    """
    
    def __init__(self, 
                 prompts_dir: Optional[str] = None,
                 default_version: str = "v1",
                 auto_reload: bool = True):
        """Initialize the PromptManager.
        
        Args:
            prompts_dir: Directory containing prompt templates
            default_version: Default version to use for templates
            auto_reload: Whether to auto-reload templates when files change
        """
        self.logger = logging.getLogger(__name__)
        
        # Set prompts directory
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            current_dir = Path(__file__).parent.parent
            self.prompts_dir = current_dir / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
            
        self.default_version = default_version
        self.auto_reload = auto_reload
        
        # Initialize Jinja2 environment
        self._setup_jinja_env()
        
        # Cache for loaded templates
        self._template_cache: Dict[str, Template] = {}
        
        self.logger.info(f"PromptManager initialized with prompts_dir: {self.prompts_dir}")
    
    def _setup_jinja_env(self) -> None:
        """Setup Jinja2 environment with custom filters and functions."""
        try:
            # Create prompts directory if it doesn't exist
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup Jinja2 environment
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                auto_reload=self.auto_reload,
                trim_blocks=True,
                lstrip_blocks=True
            )
            
            # Add custom filters
            self.jinja_env.filters['truncate_words'] = self._truncate_words
            self.jinja_env.filters['format_list'] = self._format_list
            
        except Exception as e:
            self.logger.error(f"Failed to setup Jinja2 environment: {e}")
            raise
    
    def _truncate_words(self, text: str, max_words: int = 50) -> str:
        """Custom Jinja2 filter to truncate text by word count."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + '...'
    
    def _format_list(self, items: List[str], separator: str = "\n- ") -> str:
        """Custom Jinja2 filter to format lists."""
        if not items:
            return ""
        return separator + separator.join(items)
    
    def load_template(self, template_name: str, version: Optional[str] = None) -> Template:
        """Load a prompt template.
        
        Args:
            template_name: Name of the template (without extension)
            version: Version of the template to load
            
        Returns:
            Jinja2 Template object
            
        Raises:
            TemplateNotFound: If template file doesn't exist
        """
        version = version or self.default_version
        cache_key = f"{template_name}_{version}"
        
        # Check cache first (if auto_reload is disabled)
        if not self.auto_reload and cache_key in self._template_cache:
            return self._template_cache[cache_key]
        
        # Try to load versioned template first
        template_paths = [
            f"{template_name}_{version}.txt",
            f"{version}/{template_name}.txt",
            f"{template_name}.txt"  # Fallback to unversioned
        ]
        
        template = None
        for template_path in template_paths:
            try:
                template = self.jinja_env.get_template(template_path)
                self.logger.debug(f"Loaded template: {template_path}")
                break
            except TemplateNotFound:
                continue
        
        if template is None:
            raise TemplateNotFound(f"Template '{template_name}' not found in {self.prompts_dir}")
        
        # Cache the template
        self._template_cache[cache_key] = template
        return template
    
    def render_template(self, 
                       template_name: str, 
                       variables: Dict[str, Any], 
                       version: Optional[str] = None) -> str:
        """Render a prompt template with variables.
        
        Args:
            template_name: Name of the template
            variables: Variables to substitute in the template
            version: Version of the template to use
            
        Returns:
            Rendered prompt string
        """
        try:
            template = self.load_template(template_name, version)
            rendered = template.render(**variables)
            
            self.logger.debug(f"Rendered template '{template_name}' with {len(variables)} variables")
            return rendered.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to render template '{template_name}': {e}")
            raise
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Dictionary with template information
        """
        info = {
            'name': template_name,
            'versions': [],
            'default_version': self.default_version,
            'exists': False
        }
        
        # Check for different versions
        for file_path in self.prompts_dir.glob(f"{template_name}*.txt"):
            info['exists'] = True
            if file_path.name == f"{template_name}.txt":
                info['versions'].append('default')
            else:
                # Extract version from filename
                version_part = file_path.stem.replace(template_name, '').lstrip('_')
                if version_part:
                    info['versions'].append(version_part)
        
        # Check for versioned directories
        for version_dir in self.prompts_dir.iterdir():
            if version_dir.is_dir():
                version_file = version_dir / f"{template_name}.txt"
                if version_file.exists():
                    info['exists'] = True
                    info['versions'].append(version_dir.name)
        
        return info
    
    def list_templates(self) -> List[str]:
        """List all available templates.
        
        Returns:
            List of template names
        """
        templates = set()
        
        # Find templates in root directory
        for file_path in self.prompts_dir.glob("*.txt"):
            template_name = file_path.stem
            # Remove version suffix if present
            if '_v' in template_name:
                template_name = template_name.split('_v')[0]
            templates.add(template_name)
        
        # Find templates in version directories
        for version_dir in self.prompts_dir.iterdir():
            if version_dir.is_dir():
                for file_path in version_dir.glob("*.txt"):
                    templates.add(file_path.stem)
        
        return sorted(list(templates))
    
    def validate_template(self, template_name: str, test_variables: Dict[str, Any]) -> bool:
        """Validate a template by attempting to render it with test variables.
        
        Args:
            template_name: Name of the template to validate
            test_variables: Test variables for rendering
            
        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.render_template(template_name, test_variables)
            return True
        except Exception as e:
            self.logger.error(f"Template validation failed for '{template_name}': {e}")
            return False
    
    def create_template(self, 
                       template_name: str, 
                       content: str, 
                       version: Optional[str] = None,
                       overwrite: bool = False) -> bool:
        """Create a new template file.
        
        Args:
            template_name: Name of the template
            content: Template content
            version: Version of the template
            overwrite: Whether to overwrite existing template
            
        Returns:
            True if template was created successfully
        """
        try:
            if version and version != 'default':
                # Create versioned template
                version_dir = self.prompts_dir / version
                version_dir.mkdir(exist_ok=True)
                template_path = version_dir / f"{template_name}.txt"
            else:
                template_path = self.prompts_dir / f"{template_name}.txt"
            
            if template_path.exists() and not overwrite:
                self.logger.warning(f"Template '{template_name}' already exists")
                return False
            
            template_path.write_text(content, encoding='utf-8')
            self.logger.info(f"Created template: {template_path}")
            
            # Clear cache for this template
            cache_key = f"{template_name}_{version or self.default_version}"
            self._template_cache.pop(cache_key, None)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create template '{template_name}': {e}")
            return False


# Global prompt manager instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance.
    
    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(template_name: str, 
                 variables: Dict[str, Any], 
                 version: Optional[str] = None) -> str:
    """Convenience function to render a prompt template.
    
    Args:
        template_name: Name of the template
        variables: Variables to substitute
        version: Version of the template
        
    Returns:
        Rendered prompt string
    """
    return get_prompt_manager().render_template(template_name, variables, version)