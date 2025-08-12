"""
Utilities Package - Optimized Components
"""

# Lazy imports for performance
def get_api_client():
    from . import api_client
    return api_client

def get_cache():
    from .advanced_cache import AdvancedCache
    return AdvancedCache

__version__ = "2.0.0"
