"""
Core Logic Package - Optimized Imports
High-performance cricket analysis and team generation
"""

# Lazy imports for better performance
def get_match_resolver():
    from .match_resolver import resolve_match_by_id
    return resolve_match_by_id

def get_data_aggregator():
    from .data_aggregator import aggregate_all_data
    return aggregate_all_data

def get_team_optimizer():
    from .simplified_team_optimizer import generate_world_class_ai_teams
    return generate_world_class_ai_teams

def get_learning_optimizer():
    from .learning_enhanced_optimizer import enhance_teams_with_learning
    return enhance_teams_with_learning

# Version info
__version__ = "2.0.0"
__author__ = "Dream11 AI Team"
