#!/usr/bin/env python3
"""
A/B Testing Framework for Prediction Models
Advanced testing framework for comparing prediction models and strategies
"""

import sqlite3
import json
import logging
import hashlib
import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import statistics
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status options"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class StatisticalSignificance(Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    SIGNIFICANT = "significant"  # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01

@dataclass
class TestVariant:
    """A/B test variant configuration"""
    variant_id: str
    variant_name: str
    description: str
    model_config: Dict[str, Any]
    traffic_allocation: float  # 0.0 to 1.0
    is_control: bool = False
    is_active: bool = True

@dataclass
class TestResult:
    """Individual test result data point"""
    result_id: str
    test_id: str
    variant_id: str
    user_segment: str
    predicted_score: float
    actual_score: Optional[float]
    accuracy_score: Optional[float]
    prediction_timestamp: datetime
    result_timestamp: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ABTestExperiment:
    """A/B test experiment configuration"""
    test_id: str
    test_name: str
    description: str
    variants: List[TestVariant]
    start_date: datetime
    end_date: Optional[datetime]
    status: TestStatus
    success_metric: str  # 'accuracy', 'mae', 'rmse', 'roi'
    minimum_sample_size: int
    confidence_level: float  # 0.95 for 95% confidence
    expected_effect_size: float
    created_by: str
    tags: List[str] = field(default_factory=list)

class ABTestingFramework:
    """
    Advanced A/B testing framework for prediction model comparison
    """
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        
        # Active experiments
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.variant_allocations: Dict[str, Dict[str, float]] = {}
        
        # Statistical testing parameters
        self.default_confidence_level = 0.95
        self.minimum_effect_size = 0.05  # 5% minimum detectable effect
        self.maximum_experiment_duration = 30  # days
        
        # User segmentation
        self.user_segments = {
            'new_users': 0.2,
            'regular_users': 0.6,
            'power_users': 0.2
        }
        
        # Initialize database
        self._init_database()
        
        # Load active experiments
        self._load_active_experiments()
    
    def _init_database(self):
        """Initialize A/B testing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                test_id TEXT PRIMARY KEY,
                test_name TEXT NOT NULL,
                description TEXT,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP,
                status TEXT NOT NULL,
                success_metric TEXT NOT NULL,
                minimum_sample_size INTEGER,
                confidence_level REAL,
                expected_effect_size REAL,
                created_by TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Variants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_variants (
                variant_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                description TEXT,
                model_config TEXT NOT NULL,
                traffic_allocation REAL NOT NULL,
                is_control BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (test_id) REFERENCES experiments (test_id)
            )
        ''')
        
        # Test results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                result_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                user_segment TEXT,
                predicted_score REAL NOT NULL,
                actual_score REAL,
                accuracy_score REAL,
                prediction_timestamp TIMESTAMP NOT NULL,
                result_timestamp TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (test_id) REFERENCES experiments (test_id),
                FOREIGN KEY (variant_id) REFERENCES test_variants (variant_id)
            )
        ''')
        
        # Statistical analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistical_analysis (
                analysis_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                analysis_date TIMESTAMP NOT NULL,
                variant_a_id TEXT NOT NULL,
                variant_b_id TEXT NOT NULL,
                sample_size_a INTEGER,
                sample_size_b INTEGER,
                mean_a REAL,
                mean_b REAL,
                variance_a REAL,
                variance_b REAL,
                effect_size REAL,
                p_value REAL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                significance_level TEXT,
                conclusion TEXT,
                FOREIGN KEY (test_id) REFERENCES experiments (test_id)
            )
        ''')
        
        # Experiment segments tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_segments (
                segment_id TEXT PRIMARY KEY,
                test_id TEXT NOT NULL,
                segment_name TEXT NOT NULL,
                segment_criteria TEXT,
                allocation_percentage REAL,
                active_participants INTEGER DEFAULT 0,
                FOREIGN KEY (test_id) REFERENCES experiments (test_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ A/B testing database initialized: {self.db_path}")
    
    def create_experiment(self, test_name: str, description: str, variants: List[Dict[str, Any]],
                         success_metric: str = 'accuracy', minimum_sample_size: int = 100,
                         duration_days: int = 14, created_by: str = 'system') -> str:
        """
        Create a new A/B test experiment
        
        Args:
            test_name: Name of the experiment
            description: Description of what's being tested
            variants: List of variant configurations
            success_metric: Metric to optimize ('accuracy', 'mae', 'rmse')
            minimum_sample_size: Minimum sample size per variant
            duration_days: Duration of the experiment in days
            created_by: Creator of the experiment
            
        Returns:
            test_id: Unique identifier for the experiment
        """
        # Generate unique test ID
        test_id = hashlib.sha256(
            f"{test_name}_{datetime.now().isoformat()}_{created_by}".encode()
        ).hexdigest()[:16]
        
        # Validate traffic allocation
        total_allocation = sum(variant.get('traffic_allocation', 0) for variant in variants)
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Total traffic allocation must equal 1.0, got {total_allocation}")
        
        # Create test variants
        test_variants = []
        for i, variant_config in enumerate(variants):
            variant_id = f"{test_id}_variant_{i}"
            variant = TestVariant(
                variant_id=variant_id,
                variant_name=variant_config.get('name', f'Variant {i}'),
                description=variant_config.get('description', ''),
                model_config=variant_config.get('model_config', {}),
                traffic_allocation=variant_config.get('traffic_allocation', 0),
                is_control=variant_config.get('is_control', i == 0)  # First variant is control by default
            )
            test_variants.append(variant)
        
        # Create experiment
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        experiment = ABTestExperiment(
            test_id=test_id,
            test_name=test_name,
            description=description,
            variants=test_variants,
            start_date=start_date,
            end_date=end_date,
            status=TestStatus.DRAFT,
            success_metric=success_metric,
            minimum_sample_size=minimum_sample_size,
            confidence_level=self.default_confidence_level,
            expected_effect_size=self.minimum_effect_size,
            created_by=created_by
        )
        
        # Save to database
        self._save_experiment(experiment)
        
        logger.info(f"🧪 Created A/B test experiment: {test_name} ({test_id})")
        return test_id
    
    def start_experiment(self, test_id: str) -> bool:
        """Start an A/B test experiment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE experiments 
                SET status = ?, updated_at = ?
                WHERE test_id = ?
            ''', (TestStatus.RUNNING.value, datetime.now(), test_id))
            
            conn.commit()
            conn.close()
            
            # Load into active experiments
            self._load_experiment(test_id)
            
            logger.info(f"🚀 Started A/B test experiment: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting experiment {test_id}: {e}")
            return False
    
    def assign_variant(self, test_id: str, user_id: str = None, user_segment: str = None) -> Optional[str]:
        """
        Assign a user to a test variant
        
        Args:
            test_id: ID of the experiment
            user_id: User identifier (for consistent assignment)
            user_segment: User segment for targeted testing
            
        Returns:
            variant_id: Assigned variant ID or None if experiment not active
        """
        if test_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[test_id]
        
        # Check if experiment is running
        if experiment.status != TestStatus.RUNNING:
            return None
        
        # Check if experiment has ended
        if experiment.end_date and datetime.now() > experiment.end_date:
            return None
        
        # Consistent assignment based on user_id if provided
        if user_id:
            # Use hash of user_id and test_id for consistent assignment
            hash_input = f"{user_id}_{test_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            assignment_value = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        else:
            # Random assignment
            assignment_value = random.random()
        
        # Find variant based on traffic allocation
        cumulative_allocation = 0.0
        for variant in experiment.variants:
            if not variant.is_active:
                continue
            
            cumulative_allocation += variant.traffic_allocation
            if assignment_value <= cumulative_allocation:
                return variant.variant_id
        
        # Fallback to control variant
        control_variants = [v for v in experiment.variants if v.is_control and v.is_active]
        if control_variants:
            return control_variants[0].variant_id
        
        return None
    
    def record_prediction_result(self, test_id: str, variant_id: str, predicted_score: float,
                               actual_score: Optional[float] = None, user_segment: str = 'general',
                               metadata: Dict[str, Any] = None) -> str:
        """
        Record a prediction result for A/B testing
        
        Args:
            test_id: ID of the experiment
            variant_id: ID of the variant used
            predicted_score: Predicted score
            actual_score: Actual score (if available)
            user_segment: User segment
            metadata: Additional metadata
            
        Returns:
            result_id: Unique identifier for the result
        """
        result_id = hashlib.sha256(
            f"{test_id}_{variant_id}_{datetime.now().isoformat()}_{random.random()}".encode()
        ).hexdigest()[:16]
        
        # Calculate accuracy score if actual score is available
        accuracy_score = None
        if actual_score is not None:
            if predicted_score > 0:
                accuracy_score = 1 - abs(predicted_score - actual_score) / max(predicted_score, actual_score)
            else:
                accuracy_score = 1 - abs(actual_score) / 100  # Assuming max score of 100
        
        # Create result object
        result = TestResult(
            result_id=result_id,
            test_id=test_id,
            variant_id=variant_id,
            user_segment=user_segment,
            predicted_score=predicted_score,
            actual_score=actual_score,
            accuracy_score=accuracy_score,
            prediction_timestamp=datetime.now(),
            result_timestamp=datetime.now() if actual_score is not None else None,
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_test_result(result)
        
        return result_id
    
    def analyze_experiment(self, test_id: str, force_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform statistical analysis of A/B test experiment
        
        Args:
            test_id: ID of the experiment to analyze
            force_analysis: Force analysis even with small sample size
            
        Returns:
            Analysis results including statistical significance
        """
        if test_id not in self.active_experiments and not force_analysis:
            return {'error': 'Experiment not active'}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiment details
            cursor.execute('SELECT * FROM experiments WHERE test_id = ?', (test_id,))
            experiment_data = cursor.fetchone()
            
            if not experiment_data:
                return {'error': 'Experiment not found'}
            
            # Get variants
            cursor.execute('SELECT * FROM test_variants WHERE test_id = ?', (test_id,))
            variants_data = cursor.fetchall()
            
            # Get results by variant
            cursor.execute('''
                SELECT variant_id, predicted_score, actual_score, accuracy_score
                FROM test_results 
                WHERE test_id = ? AND actual_score IS NOT NULL
            ''', (test_id,))
            results_data = cursor.fetchall()
            
            conn.close()
            
            # Organize results by variant
            variant_results = defaultdict(list)
            for result in results_data:
                variant_id, predicted_score, actual_score, accuracy_score = result
                variant_results[variant_id].append({
                    'predicted_score': predicted_score,
                    'actual_score': actual_score,
                    'accuracy_score': accuracy_score
                })
            
            # Perform statistical analysis
            analysis_results = self._perform_statistical_analysis(
                variant_results, 
                experiment_data[5],  # success_metric
                experiment_data[7]   # confidence_level
            )
            
            # Save analysis results
            self._save_analysis_results(test_id, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing experiment {test_id}: {e}")
            return {'error': str(e)}
    
    def _perform_statistical_analysis(self, variant_results: Dict[str, List[Dict]], 
                                    success_metric: str, confidence_level: float) -> Dict[str, Any]:
        """Perform statistical analysis on variant results"""
        
        analysis = {
            'analysis_date': datetime.now().isoformat(),
            'success_metric': success_metric,
            'confidence_level': confidence_level,
            'variant_statistics': {},
            'pairwise_comparisons': [],
            'recommendations': []
        }
        
        # Calculate statistics for each variant
        for variant_id, results in variant_results.items():
            if not results:
                continue
            
            # Extract metric values based on success_metric
            if success_metric == 'accuracy':
                values = [r['accuracy_score'] for r in results if r['accuracy_score'] is not None]
            elif success_metric == 'mae':
                values = [abs(r['predicted_score'] - r['actual_score']) for r in results]
            elif success_metric == 'rmse':
                values = [math.sqrt((r['predicted_score'] - r['actual_score']) ** 2) for r in results]
            else:
                values = [r['predicted_score'] for r in results]
            
            if values:
                analysis['variant_statistics'][variant_id] = {
                    'sample_size': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'confidence_interval': self._calculate_confidence_interval(values, confidence_level)
                }
        
        # Perform pairwise comparisons
        variant_ids = list(variant_results.keys())
        for i, variant_a in enumerate(variant_ids):
            for variant_b in variant_ids[i+1:]:
                comparison = self._compare_variants(
                    variant_results[variant_a],
                    variant_results[variant_b],
                    variant_a,
                    variant_b,
                    success_metric,
                    confidence_level
                )
                analysis['pairwise_comparisons'].append(comparison)
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _compare_variants(self, results_a: List[Dict], results_b: List[Dict], 
                         variant_a_id: str, variant_b_id: str, success_metric: str,
                         confidence_level: float) -> Dict[str, Any]:
        """Compare two variants using statistical tests"""
        
        # Extract values for comparison
        def extract_values(results, metric):
            if metric == 'accuracy':
                return [r['accuracy_score'] for r in results if r['accuracy_score'] is not None]
            elif metric == 'mae':
                return [abs(r['predicted_score'] - r['actual_score']) for r in results]
            elif metric == 'rmse':
                return [math.sqrt((r['predicted_score'] - r['actual_score']) ** 2) for r in results]
            else:
                return [r['predicted_score'] for r in results]
        
        values_a = extract_values(results_a, success_metric)
        values_b = extract_values(results_b, success_metric)
        
        if not values_a or not values_b:
            return {
                'variant_a': variant_a_id,
                'variant_b': variant_b_id,
                'error': 'Insufficient data for comparison'
            }
        
        # Perform t-test (simplified implementation)
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        
        if len(values_a) > 1:
            var_a = statistics.variance(values_a)
        else:
            var_a = 0
            
        if len(values_b) > 1:
            var_b = statistics.variance(values_b)
        else:
            var_b = 0
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt((var_a + var_b) / 2) if var_a + var_b > 0 else 1
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Simplified p-value calculation (this would use proper t-test in production)
        # For demonstration, using a heuristic based on effect size and sample sizes
        min_sample_size = min(len(values_a), len(values_b))
        p_value = max(0.001, 1 - abs(effect_size) * math.sqrt(min_sample_size) / 5)
        
        # Determine statistical significance
        if p_value < 0.01:
            significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value < 0.05:
            significance = StatisticalSignificance.SIGNIFICANT
        elif p_value < 0.1:
            significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT
        
        # Determine winner
        if success_metric in ['accuracy']:
            winner = variant_a_id if mean_a > mean_b else variant_b_id
        else:  # For error metrics (mae, rmse), lower is better
            winner = variant_a_id if mean_a < mean_b else variant_b_id
        
        return {
            'variant_a': variant_a_id,
            'variant_b': variant_b_id,
            'sample_size_a': len(values_a),
            'sample_size_b': len(values_b),
            'mean_a': mean_a,
            'mean_b': mean_b,
            'effect_size': effect_size,
            'p_value': p_value,
            'statistical_significance': significance.value,
            'winner': winner,
            'improvement': abs(mean_a - mean_b),
            'improvement_percentage': abs(mean_a - mean_b) / max(mean_a, mean_b, 0.01) * 100
        }
    
    def _calculate_confidence_interval(self, values: List[float], confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for a set of values"""
        if len(values) < 2:
            mean_val = values[0] if values else 0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Using normal approximation (for large samples) or t-distribution
        if len(values) >= 30:
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        else:
            # Simplified t-score approximation
            z_score = 2.0 if confidence_level == 0.95 else 2.8
        
        margin_of_error = z_score * std_err
        
        return (mean_val - margin_of_error, mean_val + margin_of_error)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Check sample sizes
        variant_stats = analysis.get('variant_statistics', {})
        min_sample_size = min([stats['sample_size'] for stats in variant_stats.values()]) if variant_stats else 0
        
        if min_sample_size < 30:
            recommendations.append("⚠️ Sample sizes are small. Continue collecting data for more reliable results.")
        
        # Check for statistically significant results
        significant_comparisons = [
            comp for comp in analysis.get('pairwise_comparisons', [])
            if comp.get('statistical_significance') in ['significant', 'highly_significant']
        ]
        
        if significant_comparisons:
            best_comparison = max(significant_comparisons, key=lambda x: x.get('improvement_percentage', 0))
            recommendations.append(
                f"✅ Variant {best_comparison['winner']} shows significant improvement "
                f"({best_comparison['improvement_percentage']:.1f}%)"
            )
        else:
            recommendations.append("📊 No statistically significant differences detected yet.")
        
        # Check for concerning patterns
        for comp in analysis.get('pairwise_comparisons', []):
            if comp.get('effect_size', 0) > 0.8:  # Large effect size
                recommendations.append(
                    f"🎯 Large effect size detected between variants - consider early decision"
                )
                break
        
        return recommendations
    
    def _load_active_experiments(self):
        """Load active experiments from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT test_id FROM experiments 
                WHERE status = ?
            ''', (TestStatus.RUNNING.value,))
            
            active_test_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            for test_id in active_test_ids:
                self._load_experiment(test_id)
            
            logger.info(f"📊 Loaded {len(active_test_ids)} active A/B experiments")
            
        except Exception as e:
            logger.error(f"❌ Error loading active experiments: {e}")
    
    def _load_experiment(self, test_id: str):
        """Load a specific experiment from database"""
        # Implementation for loading experiment details
        # This would fetch the full experiment configuration
        pass
    
    def _save_experiment(self, experiment: ABTestExperiment):
        """Save experiment to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save experiment
            cursor.execute('''
                INSERT OR REPLACE INTO experiments 
                (test_id, test_name, description, start_date, end_date, status, 
                 success_metric, minimum_sample_size, confidence_level, 
                 expected_effect_size, created_by, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment.test_id, experiment.test_name, experiment.description,
                experiment.start_date, experiment.end_date, experiment.status.value,
                experiment.success_metric, experiment.minimum_sample_size,
                experiment.confidence_level, experiment.expected_effect_size,
                experiment.created_by, json.dumps(experiment.tags)
            ))
            
            # Save variants
            for variant in experiment.variants:
                cursor.execute('''
                    INSERT OR REPLACE INTO test_variants
                    (variant_id, test_id, variant_name, description, model_config,
                     traffic_allocation, is_control, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    variant.variant_id, experiment.test_id, variant.variant_name,
                    variant.description, json.dumps(variant.model_config),
                    variant.traffic_allocation, variant.is_control, variant.is_active
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving experiment: {e}")
    
    def _save_test_result(self, result: TestResult):
        """Save test result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO test_results
                (result_id, test_id, variant_id, user_segment, predicted_score,
                 actual_score, accuracy_score, prediction_timestamp, result_timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.result_id, result.test_id, result.variant_id, result.user_segment,
                result.predicted_score, result.actual_score, result.accuracy_score,
                result.prediction_timestamp, result.result_timestamp,
                json.dumps(result.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving test result: {e}")
    
    def _save_analysis_results(self, test_id: str, analysis: Dict[str, Any]):
        """Save analysis results to database"""
        # Implementation for saving detailed analysis results
        pass
    
    def get_experiment_dashboard(self, test_id: str) -> Dict[str, Any]:
        """Get dashboard data for an experiment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get experiment overview
            cursor.execute('SELECT * FROM experiments WHERE test_id = ?', (test_id,))
            experiment_data = cursor.fetchone()
            
            if not experiment_data:
                return {'error': 'Experiment not found'}
            
            # Get variants
            cursor.execute('SELECT * FROM test_variants WHERE test_id = ?', (test_id,))
            variants_data = cursor.fetchall()
            
            # Get results summary
            cursor.execute('''
                SELECT variant_id, COUNT(*) as total_results,
                       COUNT(CASE WHEN actual_score IS NOT NULL THEN 1 END) as complete_results,
                       AVG(predicted_score) as avg_predicted,
                       AVG(actual_score) as avg_actual,
                       AVG(accuracy_score) as avg_accuracy
                FROM test_results 
                WHERE test_id = ?
                GROUP BY variant_id
            ''', (test_id,))
            results_summary = cursor.fetchall()
            
            conn.close()
            
            # Format dashboard data
            dashboard = {
                'experiment': {
                    'test_id': experiment_data[0],
                    'test_name': experiment_data[1],
                    'status': experiment_data[4],
                    'start_date': experiment_data[3],
                    'end_date': experiment_data[4]
                },
                'variants': [
                    {
                        'variant_id': v[0],
                        'variant_name': v[2],
                        'traffic_allocation': v[5],
                        'is_control': v[6]
                    }
                    for v in variants_data
                ],
                'performance_summary': {
                    variant[0]: {
                        'total_predictions': variant[1],
                        'completed_results': variant[2],
                        'completion_rate': f"{variant[2]/variant[1]*100:.1f}%" if variant[1] > 0 else "0%",
                        'avg_predicted_score': variant[3] or 0,
                        'avg_actual_score': variant[4] or 0,
                        'avg_accuracy': variant[5] or 0
                    }
                    for variant in results_summary
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"❌ Error getting dashboard data: {e}")
            return {'error': str(e)}

# Global A/B testing framework
ab_testing = ABTestingFramework()

def get_ab_testing_framework() -> ABTestingFramework:
    """Get global A/B testing framework"""
    return ab_testing

def create_prediction_experiment(test_name: str, variants: List[Dict[str, Any]], **kwargs) -> str:
    """Create A/B test experiment for prediction models"""
    return ab_testing.create_experiment(test_name, variants=variants, **kwargs)

def get_test_variant(test_id: str, user_id: str = None) -> Optional[str]:
    """Get assigned variant for A/B test"""
    return ab_testing.assign_variant(test_id, user_id)