
# AUTO-LEARNING HOOK
# Add this to your main dream11_ai.py file

def enable_auto_learning():
    '''Enable automatic learning from match results'''
    from ai_learning_system import Dream11LearningSystem
    return Dream11LearningSystem()

def log_prediction_for_learning(learning_system, match_id, teams_data, ai_strategies):
    '''Log prediction for future learning'''
    if learning_system:
        learning_system.log_prediction(match_id, teams_data, ai_strategies)

def learn_from_result(learning_system, match_id, winning_team, winning_score, predicted_teams):
    '''Learn from actual match result'''
    if learning_system:
        return learning_system.auto_learn_from_match(match_id, predicted_teams, winning_team, winning_score)

# Usage in dream11_ai.py:
# 1. learning_system = enable_auto_learning()
# 2. After generating teams: log_prediction_for_learning(learning_system, match_id, teams_data, teams)
# 3. After match completes: learn_from_result(learning_system, match_id, winning_team, score, predicted_teams)
