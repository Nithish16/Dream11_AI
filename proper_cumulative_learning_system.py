#!/usr/bin/env python3
"""
Proper Cumulative Learning System
Accumulates knowledge from ALL matches, not just the most recent one
"""

import sqlite3
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class LearningPattern:
    """Represents a learning pattern with accumulated evidence"""
    pattern_id: str
    category: str
    subcategory: str
    description: str
    evidence_count: int = 0
    success_count: int = 0
    confidence_score: float = 0.0
    contexts: Dict[str, int] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.evidence_count if self.evidence_count > 0 else 0.0
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability based on sample size and success rate"""
        if self.evidence_count < 3:
            return 0.0  # Need minimum evidence
        
        # Wilson confidence interval for reliability
        n = self.evidence_count
        p = self.success_rate
        z = 1.96  # 95% confidence
        
        denominator = 1 + (z**2 / n)
        numerator = p + (z**2 / (2*n)) - z * math.sqrt((p*(1-p) + z**2/(4*n)) / n)
        
        return max(0, numerator / denominator)

class CumulativeLearningSystem:
    """
    Proper learning system that accumulates knowledge across all matches
    """
    
    def __init__(self, db_path: str = "data/cumulative_learning.db"):
        self.db_path = db_path
        self.setup_proper_database()
        self.learning_patterns: Dict[str, LearningPattern] = {}
        self.load_existing_patterns()
    
    def setup_proper_database(self):
        """Setup proper database schema for cumulative learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Learning patterns table - stores accumulated evidence
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL,
                description TEXT NOT NULL,
                evidence_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                contexts TEXT DEFAULT '{}',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Match evidence table - stores individual match evidence
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                context_data TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                evidence_strength REAL DEFAULT 1.0,
                match_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES learning_patterns(pattern_id)
            )
        ''')
        
        # Context clusters table - groups similar match conditions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context_clusters (
                cluster_id TEXT PRIMARY KEY,
                cluster_name TEXT NOT NULL,
                match_format TEXT,
                team_types TEXT,
                venue_type TEXT,
                conditions TEXT,
                pattern_weights TEXT DEFAULT '{}'
            )
        ''')
        
        # Prediction performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patterns_used TEXT NOT NULL,
                prediction_accuracy REAL,
                confidence_level REAL,
                actual_outcome TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Proper cumulative learning database initialized")
    
    def load_existing_patterns(self):
        """Load all existing learning patterns from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM learning_patterns')
        rows = cursor.fetchall()
        
        for row in rows:
            pattern = LearningPattern(
                pattern_id=row[0],
                category=row[1], 
                subcategory=row[2],
                description=row[3],
                evidence_count=row[4],
                success_count=row[5],
                confidence_score=row[6],
                contexts=json.loads(row[7]) if row[7] else {},
                last_updated=row[9]
            )
            self.learning_patterns[pattern.pattern_id] = pattern
        
        conn.close()
        print(f"✅ Loaded {len(self.learning_patterns)} existing learning patterns")
    
    def add_match_evidence(self, match_id: str, match_analysis: Dict[str, Any]):
        """Add evidence from a match to accumulate learning"""
        print(f"📊 Adding match evidence for {match_id} to cumulative learning...")
        
        # Extract evidence patterns from match analysis
        evidence_patterns = self._extract_evidence_patterns(match_analysis)
        
        for pattern_data in evidence_patterns:
            pattern_id = pattern_data['pattern_id']
            
            # Update or create learning pattern
            if pattern_id in self.learning_patterns:
                self._update_existing_pattern(pattern_id, pattern_data, match_id)
            else:
                self._create_new_pattern(pattern_id, pattern_data, match_id)
        
        # Update database
        self._save_patterns_to_database()
        self._save_match_evidence(match_id, evidence_patterns)
        
        print(f"✅ Added evidence from match {match_id} to {len(evidence_patterns)} patterns")
    
    def _extract_evidence_patterns(self, match_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract learning patterns from match analysis"""
        patterns = []
        
        # Captain performance patterns
        if 'captain_analysis' in match_analysis:
            cap_analysis = match_analysis['captain_analysis']
            best_captain = cap_analysis.get('best_actual_captain')
            
            if best_captain:
                patterns.append({
                    'pattern_id': f"captain_effectiveness_{best_captain.lower().replace(' ', '_')}",
                    'category': 'captain_selection',
                    'subcategory': 'player_specific',
                    'description': f"{best_captain} performs well as captain",
                    'success': True,
                    'context': {
                        'format': match_analysis.get('format', 'unknown'),
                        'conditions': match_analysis.get('conditions', 'unknown')
                    },
                    'evidence_strength': cap_analysis.get('captain_accuracy_score', 0) / 100.0
                })
        
        # Team balance patterns
        if 'team_balance_analysis' in match_analysis:
            balance = match_analysis['team_balance_analysis']
            best_balance = balance.get('best_balance_type')
            
            if best_balance:
                patterns.append({
                    'pattern_id': f"team_balance_{best_balance}",
                    'category': 'team_composition',
                    'subcategory': 'balance_strategy',
                    'description': f"Teams with {best_balance} approach are effective",
                    'success': True,
                    'context': {
                        'format': match_analysis.get('format', 'unknown'),
                        'match_type': match_analysis.get('match_type', 'unknown')
                    },
                    'evidence_strength': 0.8
                })
        
        # Venue/conditions patterns
        if 'venue_weather_analysis' in match_analysis:
            venue = match_analysis['venue_weather_analysis']
            actual_pattern = venue.get('actual_match_pattern')
            
            if actual_pattern:
                patterns.append({
                    'pattern_id': f"venue_pattern_{actual_pattern}",
                    'category': 'venue_analysis',
                    'subcategory': 'match_conditions',
                    'description': f"Venue shows {actual_pattern} characteristics",
                    'success': venue.get('prediction_accuracy', False),
                    'context': {
                        'venue': match_analysis.get('venue', 'unknown'),
                        'conditions': actual_pattern
                    },
                    'evidence_strength': 0.7
                })
        
        # Player form patterns
        if 'player_selection_analysis' in match_analysis:
            player_analysis = match_analysis['player_selection_analysis']
            
            for player, performance in player_analysis.get('selected_player_performances', {}).items():
                if performance.get('performance_score', 0) > 50:  # Good performance threshold
                    patterns.append({
                        'pattern_id': f"player_form_{player.lower().replace(' ', '_')}",
                        'category': 'player_selection',
                        'subcategory': 'individual_performance',
                        'description': f"{player} shows consistent good form",
                        'success': True,
                        'context': {
                            'format': match_analysis.get('format', 'unknown'),
                            'role': performance.get('role', 'unknown')
                        },
                        'evidence_strength': min(1.0, performance.get('performance_score', 0) / 100.0)
                    })
        
        return patterns
    
    def _update_existing_pattern(self, pattern_id: str, pattern_data: Dict, match_id: str):
        """Update existing learning pattern with new evidence"""
        pattern = self.learning_patterns[pattern_id]
        
        # Increment evidence count
        pattern.evidence_count += 1
        
        # Update success count if this evidence supports the pattern
        if pattern_data.get('success', False):
            pattern.success_count += 1
        
        # Update contexts
        context_key = f"{pattern_data.get('context', {}).get('format', 'unknown')}_" + \
                     f"{pattern_data.get('context', {}).get('conditions', 'unknown')}"
        
        pattern.contexts[context_key] = pattern.contexts.get(context_key, 0) + 1
        
        # Recalculate confidence score
        pattern.confidence_score = self._calculate_confidence_score(pattern)
        pattern.last_updated = datetime.now().isoformat()
    
    def _create_new_pattern(self, pattern_id: str, pattern_data: Dict, match_id: str):
        """Create new learning pattern"""
        context_key = f"{pattern_data.get('context', {}).get('format', 'unknown')}_" + \
                     f"{pattern_data.get('context', {}).get('conditions', 'unknown')}"
        
        pattern = LearningPattern(
            pattern_id=pattern_id,
            category=pattern_data.get('category', 'general'),
            subcategory=pattern_data.get('subcategory', 'general'),
            description=pattern_data.get('description', 'Pattern'),
            evidence_count=1,
            success_count=1 if pattern_data.get('success', False) else 0,
            contexts={context_key: 1}
        )
        
        pattern.confidence_score = self._calculate_confidence_score(pattern)
        self.learning_patterns[pattern_id] = pattern
    
    def _calculate_confidence_score(self, pattern: LearningPattern) -> float:
        """Calculate confidence score based on evidence and success rate"""
        if pattern.evidence_count < 2:
            return 0.1  # Very low confidence for single evidence
        
        # Base confidence from success rate
        success_confidence = pattern.success_rate
        
        # Sample size confidence (more evidence = higher confidence)
        sample_confidence = min(1.0, pattern.evidence_count / 10.0)  # Max confidence at 10+ samples
        
        # Context diversity bonus (patterns that work across contexts are more reliable)
        context_diversity = min(1.0, len(pattern.contexts) / 3.0)
        
        # Combined confidence score
        return (success_confidence * 0.5) + (sample_confidence * 0.3) + (context_diversity * 0.2)
    
    def get_accumulated_insights(self, context: Dict[str, Any] = None) -> Dict[str, List[LearningPattern]]:
        """Get accumulated insights across all matches, filtered by context if provided"""
        insights = defaultdict(list)
        
        for pattern in self.learning_patterns.values():
            # Filter by context if provided
            if context:
                if not self._matches_context(pattern, context):
                    continue
            
            # Only include patterns with reasonable confidence
            if pattern.confidence_score >= 0.3 and pattern.evidence_count >= 2:
                insights[pattern.category].append(pattern)
        
        # Sort by reliability score within each category
        for category in insights:
            insights[category].sort(key=lambda p: p.reliability_score, reverse=True)
        
        return dict(insights)
    
    def _matches_context(self, pattern: LearningPattern, context: Dict[str, Any]) -> bool:
        """Check if pattern matches given context"""
        format_match = context.get('format', '').lower()
        venue_match = context.get('venue', '').lower()
        
        # Check if any of the pattern's contexts match the query context
        for context_key in pattern.contexts:
            if format_match in context_key.lower():
                return True
            if venue_match in context_key.lower():
                return True
        
        return True  # If no specific context matching, include the pattern
    
    def _save_patterns_to_database(self):
        """Save all patterns to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in self.learning_patterns.values():
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns 
                (pattern_id, category, subcategory, description, evidence_count, 
                 success_count, confidence_score, contexts, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id, pattern.category, pattern.subcategory,
                pattern.description, pattern.evidence_count, pattern.success_count,
                pattern.confidence_score, json.dumps(pattern.contexts),
                pattern.last_updated
            ))
        
        conn.commit()
        conn.close()
    
    def _save_match_evidence(self, match_id: str, evidence_patterns: List[Dict[str, Any]]):
        """Save individual match evidence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern_data in evidence_patterns:
            cursor.execute('''
                INSERT INTO match_evidence 
                (match_id, pattern_id, context_data, evidence_type, success, evidence_strength)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                match_id, pattern_data['pattern_id'],
                json.dumps(pattern_data.get('context', {})),
                pattern_data.get('category', 'general'),
                pattern_data.get('success', False),
                pattern_data.get('evidence_strength', 1.0)
            ))
        
        conn.commit()
        conn.close()
    
    def generate_prediction_weights(self, match_context: Dict[str, Any]) -> Dict[str, float]:
        """Generate prediction weights based on accumulated learning"""
        accumulated_insights = self.get_accumulated_insights(match_context)
        weights = defaultdict(float)
        
        # Captain selection weights
        for pattern in accumulated_insights.get('captain_selection', []):
            if pattern.reliability_score > 0.5:
                weights[f"captain_{pattern.pattern_id}"] = pattern.reliability_score
        
        # Team balance weights
        for pattern in accumulated_insights.get('team_composition', []):
            if pattern.reliability_score > 0.4:
                weights[f"balance_{pattern.pattern_id}"] = pattern.reliability_score
        
        # Player selection weights
        for pattern in accumulated_insights.get('player_selection', []):
            if pattern.reliability_score > 0.3:
                weights[f"player_{pattern.pattern_id}"] = pattern.reliability_score
        
        return dict(weights)
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of accumulated learning"""
        total_patterns = len(self.learning_patterns)
        reliable_patterns = sum(1 for p in self.learning_patterns.values() 
                              if p.reliability_score > 0.5)
        
        category_breakdown = defaultdict(int)
        for pattern in self.learning_patterns.values():
            category_breakdown[pattern.category] += 1
        
        return {
            'total_patterns': total_patterns,
            'reliable_patterns': reliable_patterns,
            'reliability_rate': reliable_patterns / total_patterns if total_patterns > 0 else 0,
            'category_breakdown': dict(category_breakdown),
            'top_patterns': sorted(
                self.learning_patterns.values(),
                key=lambda p: p.reliability_score,
                reverse=True
            )[:10]
        }

def main():
    """Test the cumulative learning system"""
    learning_system = CumulativeLearningSystem()
    
    # Example: Add evidence from match analysis
    sample_analysis = {
        'format': 'ODI',
        'venue': 'Cazalys Stadium',
        'captain_analysis': {
            'best_actual_captain': 'Travis Head',
            'captain_accuracy_score': 85
        },
        'team_balance_analysis': {
            'best_balance_type': 'bowling_heavy'
        },
        'venue_weather_analysis': {
            'actual_match_pattern': 'balanced',
            'prediction_accuracy': False
        }
    }
    
    learning_system.add_match_evidence('117008', sample_analysis)
    
    # Get accumulated insights
    insights = learning_system.get_accumulated_insights({'format': 'ODI'})
    
    print("\n🧠 ACCUMULATED INSIGHTS:")
    for category, patterns in insights.items():
        print(f"\n{category.upper()}:")
        for pattern in patterns[:3]:  # Top 3 per category
            print(f"  • {pattern.description}")
            print(f"    Reliability: {pattern.reliability_score:.2f} ({pattern.evidence_count} matches)")
    
    # Get learning summary
    summary = learning_system.get_learning_summary()
    print(f"\n📊 LEARNING SUMMARY:")
    print(f"Total Patterns: {summary['total_patterns']}")
    print(f"Reliable Patterns: {summary['reliable_patterns']}")
    print(f"Reliability Rate: {summary['reliability_rate']:.1%}")

if __name__ == "__main__":
    main()