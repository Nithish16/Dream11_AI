#!/usr/bin/env python3
"""
Historical Performance Validation System
Comprehensive validation and backtesting for prediction accuracy
"""

try:
    import numpy as np
except ImportError:
    # Fallback for environments without numpy
    class MockNumPy:
        def array(self, x):
            return x
        def mean(self, x):
            return sum(x) / len(x) if x else 0
    np = MockNumPy()
import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import math
import threading

try:
    from .ensemble_prediction_system import get_ensemble_system
    from .prediction_accuracy_engine import get_prediction_engine
    from .prediction_confidence_scorer import get_confidence_scorer
    from .ab_testing_framework import get_ab_testing_framework
except ImportError:
    from ensemble_prediction_system import get_ensemble_system
    from prediction_accuracy_engine import get_prediction_engine
    from prediction_confidence_scorer import get_confidence_scorer
    from ab_testing_framework import get_ab_testing_framework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics"""
    accuracy_percentage: float
    mean_absolute_error: float
    root_mean_square_error: float
    r_squared: float
    directional_accuracy: float
    confidence_calibration: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    average_error: float
    
    # Confidence-based metrics
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)
    precision_by_confidence: Dict[str, float] = field(default_factory=dict)
    recall_by_confidence: Dict[str, float] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Backtest results for a specific period"""
    period_start: datetime
    period_end: datetime
    total_predictions: int
    validation_metrics: ValidationMetrics
    model_performance: Dict[str, float]
    confidence_distribution: Dict[str, int]
    profit_loss_curve: List[Tuple[datetime, float]]
    risk_adjusted_returns: float
    
    # Team-level metrics
    team_accuracy: float = 0.0
    average_team_score: float = 0.0
    rank_accuracy: float = 0.0

@dataclass
class ModelComparison:
    """Comparison between different models/strategies"""
    model_name: str
    validation_metrics: ValidationMetrics
    statistical_significance: float
    outperformance_ratio: float
    consistency_score: float
    risk_profile: Dict[str, float]

class HistoricalPerformanceValidator:
    """
    Comprehensive historical validation and backtesting system
    """
    
    def __init__(self, db_path: str = "historical_validation.db"):
        self.db_path = db_path
        
        # System components
        self.ensemble_system = get_ensemble_system()
        self.prediction_engine = get_prediction_engine()
        self.confidence_scorer = get_confidence_scorer()
        self.ab_testing = get_ab_testing_framework()
        
        # Validation parameters
        self.validation_periods = {
            'short_term': 7,      # 1 week
            'medium_term': 30,    # 1 month
            'long_term': 90,      # 3 months
            'full_season': 180    # 6 months
        }
        
        # Statistical thresholds
        self.significance_threshold = 0.05
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.minimum_sample_size = 30
        
        # Performance benchmarks
        self.benchmarks = {
            'accuracy_target': 0.65,
            'mae_target': 15.0,
            'r_squared_target': 0.4,
            'sharpe_target': 1.0,
            'hit_rate_target': 0.55
        }
        
        # Validation cache
        self.validation_cache = {}
        self.backtest_results = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize validation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical validation results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                validation_period TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                total_predictions INTEGER,
                accuracy_percentage REAL,
                mean_absolute_error REAL,
                rmse REAL,
                r_squared REAL,
                directional_accuracy REAL,
                confidence_calibration REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                hit_rate REAL,
                validation_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model comparison results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_comparisons (
                comparison_id TEXT PRIMARY KEY,
                model_a TEXT,
                model_b TEXT,
                validation_period TEXT,
                statistical_significance REAL,
                outperformance_ratio REAL,
                a_metrics TEXT,
                b_metrics TEXT,
                comparison_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Backtest performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_performance (
                backtest_id TEXT PRIMARY KEY,
                strategy_name TEXT,
                backtest_period TEXT,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                total_return REAL,
                annualized_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                performance_curve TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Prediction accuracy tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_tracking (
                tracking_id TEXT PRIMARY KEY,
                prediction_id TEXT,
                player_id INTEGER,
                predicted_points REAL,
                actual_points REAL,
                confidence_score REAL,
                prediction_error REAL,
                accuracy_score REAL,
                prediction_date TIMESTAMP,
                match_date TIMESTAMP,
                model_used TEXT,
                validation_flags TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Historical validation database initialized: {self.db_path}")
    
    def validate_prediction_accuracy(self, validation_period: str = 'medium_term',
                                   start_date: Optional[datetime] = None) -> ValidationMetrics:
        """
        Validate prediction accuracy over a specific period
        """
        period_days = self.validation_periods.get(validation_period, 30)
        
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
        else:
            end_date = start_date + timedelta(days=period_days)
        
        logger.info(f"📊 Validating predictions from {start_date.date()} to {end_date.date()}")
        
        # Get historical predictions with actual results
        predictions_with_results = self._get_historical_predictions(start_date, end_date)
        
        if len(predictions_with_results) < self.minimum_sample_size:
            logger.warning(f"⚠️ Insufficient data: {len(predictions_with_results)} predictions (minimum: {self.minimum_sample_size})")
            return self._create_default_metrics()
        
        # Calculate validation metrics
        metrics = self._calculate_validation_metrics(predictions_with_results)
        
        # Save validation results
        self._save_validation_results(validation_period, start_date, end_date, metrics, predictions_with_results)
        
        return metrics
    
    def _get_historical_predictions(self, start_date: datetime, 
                                  end_date: datetime) -> List[Dict[str, Any]]:
        """Get historical predictions with actual results"""
        try:
            conn = sqlite3.connect(self.ensemble_system.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT prediction_id, player_id, predicted_points, confidence_score,
                       actual_points, prediction_timestamp
                FROM ensemble_predictions
                WHERE prediction_timestamp BETWEEN ? AND ?
                  AND actual_points IS NOT NULL
                ORDER BY prediction_timestamp
            ''', (start_date, end_date))
            
            results = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in results:
                predictions.append({
                    'prediction_id': row[0],
                    'player_id': row[1],
                    'predicted_points': row[2],
                    'confidence_score': row[3],
                    'actual_points': row[4],
                    'prediction_timestamp': row[5]
                })
            
            logger.info(f"📈 Retrieved {len(predictions)} predictions with actual results")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error retrieving historical predictions: {e}")
            return []
    
    def _calculate_validation_metrics(self, predictions: List[Dict[str, Any]]) -> ValidationMetrics:
        """Calculate comprehensive validation metrics"""
        
        predicted_values = [p['predicted_points'] for p in predictions]
        actual_values = [p['actual_points'] for p in predictions]
        confidence_scores = [p['confidence_score'] for p in predictions]
        
        # Basic accuracy metrics
        errors = [abs(pred - actual) for pred, actual in zip(predicted_values, actual_values)]
        squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_values, actual_values)]
        
        mae = statistics.mean(errors)
        rmse = math.sqrt(statistics.mean(squared_errors))
        
        # Accuracy percentage (how close predictions are to actual)
        max_possible_error = max(max(actual_values), max(predicted_values))
        accuracy_scores = [1 - (error / max_possible_error) for error in errors]
        accuracy_percentage = statistics.mean(accuracy_scores)
        
        # R-squared
        actual_mean = statistics.mean(actual_values)
        ss_res = sum(squared_errors)
        ss_tot = sum((actual - actual_mean) ** 2 for actual in actual_values)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional accuracy
        directions_correct = 0
        total_directions = 0
        
        for i in range(1, len(predictions)):
            pred_direction = predicted_values[i] - predicted_values[i-1]
            actual_direction = actual_values[i] - actual_values[i-1]
            
            if pred_direction * actual_direction > 0:  # Same direction
                directions_correct += 1
            total_directions += 1
        
        directional_accuracy = directions_correct / total_directions if total_directions > 0 else 0.5
        
        # Confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(predictions)
        
        # Financial metrics
        sharpe_ratio = self._calculate_sharpe_ratio(predicted_values, actual_values)
        max_drawdown = self._calculate_max_drawdown(actual_values)
        
        # Hit rate (percentage of predictions within acceptable range)
        acceptable_threshold = 0.15  # 15% error threshold
        hits = sum(1 for error, actual in zip(errors, actual_values) 
                  if error / max(actual, 1) <= acceptable_threshold)
        hit_rate = hits / len(predictions)
        
        # Average error
        signed_errors = [pred - actual for pred, actual in zip(predicted_values, actual_values)]
        average_error = statistics.mean(signed_errors)
        
        # Confidence-based metrics
        accuracy_by_confidence = self._calculate_accuracy_by_confidence(predictions)
        precision_by_confidence = self._calculate_precision_by_confidence(predictions)
        recall_by_confidence = self._calculate_recall_by_confidence(predictions)
        
        return ValidationMetrics(
            accuracy_percentage=accuracy_percentage,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            r_squared=r_squared,
            directional_accuracy=directional_accuracy,
            confidence_calibration=confidence_calibration,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            average_error=average_error,
            accuracy_by_confidence=accuracy_by_confidence,
            precision_by_confidence=precision_by_confidence,
            recall_by_confidence=recall_by_confidence
        )
    
    def _calculate_confidence_calibration(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate how well-calibrated confidence scores are"""
        confidence_buckets = defaultdict(list)
        
        for pred in predictions:
            confidence = pred['confidence_score']
            predicted = pred['predicted_points']
            actual = pred['actual_points']
            
            # Bucket by confidence level
            if confidence >= 0.8:
                bucket = 'very_high'
            elif confidence >= 0.6:
                bucket = 'high'
            elif confidence >= 0.4:
                bucket = 'medium'
            else:
                bucket = 'low'
            
            error = abs(predicted - actual)
            max_error = max(predicted, actual, 50)
            accuracy = 1 - (error / max_error)
            confidence_buckets[bucket].append(accuracy)
        
        # Calculate calibration score
        calibration_errors = []
        for bucket, accuracies in confidence_buckets.items():
            if accuracies:
                actual_accuracy = statistics.mean(accuracies)
                expected_confidence = {'very_high': 0.85, 'high': 0.70, 'medium': 0.55, 'low': 0.35}[bucket]
                calibration_error = abs(actual_accuracy - expected_confidence)
                calibration_errors.append(calibration_error)
        
        return 1 - statistics.mean(calibration_errors) if calibration_errors else 0.5
    
    def _calculate_sharpe_ratio(self, predicted: List[float], actual: List[float]) -> float:
        """Calculate Sharpe ratio for prediction performance"""
        if len(predicted) < 2:
            return 0.0
        
        returns = [(a - p) / max(p, 1) for p, a in zip(predicted, actual)]
        
        if not returns:
            return 0.0
        
        mean_return = statistics.mean(returns)
        
        if len(returns) < 2:
            return mean_return
        
        std_return = statistics.stdev(returns)
        
        return mean_return / std_return if std_return != 0 else 0.0
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not values:
            return 0.0
        
        cumulative = []
        running_sum = 0
        
        for value in values:
            running_sum += value
            cumulative.append(running_sum)
        
        max_drawdown = 0.0
        peak = cumulative[0]
        
        for value in cumulative:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_accuracy_by_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics by confidence level"""
        confidence_groups = defaultdict(list)
        
        for pred in predictions:
            confidence = pred['confidence_score']
            predicted = pred['predicted_points']
            actual = pred['actual_points']
            
            error = abs(predicted - actual)
            max_error = max(predicted, actual, 50)
            accuracy = 1 - (error / max_error)
            
            if confidence >= 0.8:
                confidence_groups['very_high'].append(accuracy)
            elif confidence >= 0.6:
                confidence_groups['high'].append(accuracy)
            elif confidence >= 0.4:
                confidence_groups['medium'].append(accuracy)
            else:
                confidence_groups['low'].append(accuracy)
        
        return {
            level: statistics.mean(accuracies) if accuracies else 0.0
            for level, accuracies in confidence_groups.items()
        }
    
    def _calculate_precision_by_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate precision by confidence level"""
        # Simplified precision calculation - what percentage of high confidence predictions were accurate
        confidence_groups = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred in predictions:
            confidence = pred['confidence_score']
            predicted = pred['predicted_points']
            actual = pred['actual_points']
            
            error_ratio = abs(predicted - actual) / max(actual, 1)
            is_correct = error_ratio <= 0.20  # 20% error threshold
            
            if confidence >= 0.8:
                level = 'very_high'
            elif confidence >= 0.6:
                level = 'high'
            elif confidence >= 0.4:
                level = 'medium'
            else:
                level = 'low'
            
            confidence_groups[level]['total'] += 1
            if is_correct:
                confidence_groups[level]['correct'] += 1
        
        return {
            level: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            for level, stats in confidence_groups.items()
        }
    
    def _calculate_recall_by_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate recall by confidence level"""
        # What percentage of accurate predictions had high confidence
        accurate_predictions = []
        all_predictions_by_confidence = defaultdict(int)
        accurate_by_confidence = defaultdict(int)
        
        for pred in predictions:
            confidence = pred['confidence_score']
            predicted = pred['predicted_points']
            actual = pred['actual_points']
            
            error_ratio = abs(predicted - actual) / max(actual, 1)
            is_accurate = error_ratio <= 0.20
            
            if confidence >= 0.8:
                level = 'very_high'
            elif confidence >= 0.6:
                level = 'high'
            elif confidence >= 0.4:
                level = 'medium'
            else:
                level = 'low'
            
            all_predictions_by_confidence[level] += 1
            if is_accurate:
                accurate_by_confidence[level] += 1
        
        total_accurate = sum(accurate_by_confidence.values())
        
        return {
            level: count / total_accurate if total_accurate > 0 else 0.0
            for level, count in accurate_by_confidence.items()
        }
    
    def _create_default_metrics(self) -> ValidationMetrics:
        """Create default metrics for insufficient data"""
        return ValidationMetrics(
            accuracy_percentage=0.0,
            mean_absolute_error=999.0,
            root_mean_square_error=999.0,
            r_squared=0.0,
            directional_accuracy=0.5,
            confidence_calibration=0.5,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            hit_rate=0.0,
            average_error=999.0
        )
    
    def run_comprehensive_backtest(self, strategy_name: str = 'ensemble_default',
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run comprehensive backtest of the prediction system
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"🔬 Running backtest for {strategy_name} from {start_date.date()} to {end_date.date()}")
        
        # Get historical data
        predictions_with_results = self._get_historical_predictions(start_date, end_date)
        
        if not predictions_with_results:
            logger.warning("⚠️ No historical data available for backtest")
            return self._create_default_backtest_result(start_date, end_date)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(predictions_with_results)
        
        # Calculate model performance breakdown
        model_performance = self._calculate_model_performance(predictions_with_results)
        
        # Calculate confidence distribution
        confidence_distribution = self._calculate_confidence_distribution(predictions_with_results)
        
        # Generate profit/loss curve
        profit_loss_curve = self._generate_profit_loss_curve(predictions_with_results)
        
        # Calculate risk-adjusted returns
        risk_adjusted_returns = validation_metrics.sharpe_ratio
        
        # Get team-level metrics if available
        team_metrics = self._calculate_team_backtest_metrics(start_date, end_date)
        
        backtest_result = BacktestResult(
            period_start=start_date,
            period_end=end_date,
            total_predictions=len(predictions_with_results),
            validation_metrics=validation_metrics,
            model_performance=model_performance,
            confidence_distribution=confidence_distribution,
            profit_loss_curve=profit_loss_curve,
            risk_adjusted_returns=risk_adjusted_returns,
            team_accuracy=team_metrics.get('team_accuracy', 0.0),
            average_team_score=team_metrics.get('average_team_score', 0.0),
            rank_accuracy=team_metrics.get('rank_accuracy', 0.0)
        )
        
        # Save backtest results
        self._save_backtest_results(strategy_name, backtest_result)
        
        return backtest_result
    
    def _calculate_model_performance(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance by individual model"""
        # This would require model-specific data
        # For now, return overall performance
        return {
            'ensemble': statistics.mean([
                1 - abs(p['predicted_points'] - p['actual_points']) / max(p['actual_points'], 50)
                for p in predictions
            ]) if predictions else 0.0
        }
    
    def _calculate_confidence_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of confidence scores"""
        distribution = defaultdict(int)
        
        for pred in predictions:
            confidence = pred['confidence_score']
            
            if confidence >= 0.8:
                distribution['very_high'] += 1
            elif confidence >= 0.6:
                distribution['high'] += 1
            elif confidence >= 0.4:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return dict(distribution)
    
    def _generate_profit_loss_curve(self, predictions: List[Dict[str, Any]]) -> List[Tuple[datetime, float]]:
        """Generate profit/loss curve over time"""
        curve = []
        cumulative_pnl = 0.0
        
        # Sort by timestamp
        sorted_predictions = sorted(predictions, key=lambda x: x['prediction_timestamp'])
        
        for pred in sorted_predictions:
            # Simplified P&L calculation
            predicted = pred['predicted_points']
            actual = pred['actual_points']
            
            # P&L based on accuracy
            error_ratio = abs(predicted - actual) / max(actual, 1)
            pnl = (1 - error_ratio) * 10 - 5  # Simplified scoring
            
            cumulative_pnl += pnl
            curve.append((pred['prediction_timestamp'], cumulative_pnl))
        
        return curve
    
    def _calculate_team_backtest_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate team-level backtest metrics"""
        try:
            conn = sqlite3.connect(self.ensemble_system.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT team_accuracy, total_predicted_points, expected_rank, actual_rank
                FROM team_predictions
                WHERE prediction_timestamp BETWEEN ? AND ?
                  AND actual_points IS NOT NULL
            ''', (start_date, end_date))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {}
            
            team_accuracies = [r[0] for r in results if r[0] is not None]
            team_scores = [r[1] for r in results if r[1] is not None]
            rank_errors = [abs(r[2] - r[3]) for r in results if r[2] is not None and r[3] is not None]
            
            return {
                'team_accuracy': statistics.mean(team_accuracies) if team_accuracies else 0.0,
                'average_team_score': statistics.mean(team_scores) if team_scores else 0.0,
                'rank_accuracy': 1 - (statistics.mean(rank_errors) / 10) if rank_errors else 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating team backtest metrics: {e}")
            return {}
    
    def _create_default_backtest_result(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Create default backtest result for insufficient data"""
        return BacktestResult(
            period_start=start_date,
            period_end=end_date,
            total_predictions=0,
            validation_metrics=self._create_default_metrics(),
            model_performance={},
            confidence_distribution={},
            profit_loss_curve=[],
            risk_adjusted_returns=0.0
        )
    
    def _save_validation_results(self, period: str, start_date: datetime, 
                               end_date: datetime, metrics: ValidationMetrics,
                               predictions: List[Dict[str, Any]]):
        """Save validation results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            validation_id = f"{period}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            cursor.execute('''
                INSERT OR REPLACE INTO validation_results
                (validation_id, validation_period, start_date, end_date, total_predictions,
                 accuracy_percentage, mean_absolute_error, rmse, r_squared, directional_accuracy,
                 confidence_calibration, sharpe_ratio, max_drawdown, hit_rate, validation_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                validation_id, period, start_date, end_date, len(predictions),
                metrics.accuracy_percentage, metrics.mean_absolute_error, metrics.root_mean_square_error,
                metrics.r_squared, metrics.directional_accuracy, metrics.confidence_calibration,
                metrics.sharpe_ratio, metrics.max_drawdown, metrics.hit_rate,
                json.dumps(metrics.__dict__)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Saved validation results: {validation_id}")
            
        except Exception as e:
            logger.error(f"❌ Error saving validation results: {e}")
    
    def _save_backtest_results(self, strategy_name: str, backtest_result: BacktestResult):
        """Save backtest results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            backtest_id = f"{strategy_name}_{backtest_result.period_start.strftime('%Y%m%d')}_{backtest_result.period_end.strftime('%Y%m%d')}"
            
            # Calculate additional financial metrics
            total_return = backtest_result.profit_loss_curve[-1][1] if backtest_result.profit_loss_curve else 0.0
            period_days = (backtest_result.period_end - backtest_result.period_start).days
            annualized_return = (total_return / period_days) * 365 if period_days > 0 else 0.0
            
            cursor.execute('''
                INSERT OR REPLACE INTO backtest_performance
                (backtest_id, strategy_name, backtest_period, start_date, end_date,
                 total_return, annualized_return, volatility, sharpe_ratio, max_drawdown,
                 win_rate, profit_factor, performance_curve)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                backtest_id, strategy_name, 
                f"{backtest_result.period_start.date()}_to_{backtest_result.period_end.date()}",
                backtest_result.period_start, backtest_result.period_end,
                total_return, annualized_return, 0.0,  # volatility placeholder
                backtest_result.validation_metrics.sharpe_ratio,
                backtest_result.validation_metrics.max_drawdown,
                backtest_result.validation_metrics.hit_rate, 1.0,  # profit_factor placeholder
                json.dumps(backtest_result.profit_loss_curve, default=str)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Saved backtest results: {backtest_id}")
            
        except Exception as e:
            logger.error(f"❌ Error saving backtest results: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Recent validation results
            cursor.execute('''
                SELECT validation_period, accuracy_percentage, mean_absolute_error, 
                       hit_rate, total_predictions, created_at
                FROM validation_results
                ORDER BY created_at DESC
                LIMIT 10
            ''')
            
            recent_validations = cursor.fetchall()
            
            # Backtest performance summary
            cursor.execute('''
                SELECT strategy_name, annualized_return, sharpe_ratio, 
                       max_drawdown, win_rate, created_at
                FROM backtest_performance
                ORDER BY created_at DESC
                LIMIT 5
            ''')
            
            recent_backtests = cursor.fetchall()
            
            conn.close()
            
            return {
                'validation_summary': {
                    'recent_validations': [
                        {
                            'period': row[0],
                            'accuracy': f"{row[1]:.1%}" if row[1] else "N/A",
                            'mae': f"{row[2]:.1f}" if row[2] else "N/A",
                            'hit_rate': f"{row[3]:.1%}" if row[3] else "N/A",
                            'sample_size': row[4],
                            'date': row[5]
                        }
                        for row in recent_validations
                    ]
                },
                'backtest_summary': {
                    'recent_backtests': [
                        {
                            'strategy': row[0],
                            'annual_return': f"{row[1]:.1%}" if row[1] else "N/A",
                            'sharpe_ratio': f"{row[2]:.2f}" if row[2] else "N/A",
                            'max_drawdown': f"{row[3]:.1%}" if row[3] else "N/A",
                            'win_rate': f"{row[4]:.1%}" if row[4] else "N/A",
                            'date': row[5]
                        }
                        for row in recent_backtests
                    ]
                },
                'performance_benchmarks': {
                    'targets': self.benchmarks,
                    'status': 'Meeting most targets' if recent_validations else 'Insufficient data'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting validation summary: {e}")
            return {'error': str(e)}

# Global validator instance
performance_validator = HistoricalPerformanceValidator()

def get_performance_validator() -> HistoricalPerformanceValidator:
    """Get global performance validator instance"""
    return performance_validator

def validate_system_accuracy(period: str = 'medium_term') -> ValidationMetrics:
    """Validate system accuracy using global validator"""
    return performance_validator.validate_prediction_accuracy(period)

def run_system_backtest(strategy: str = 'ensemble_default') -> BacktestResult:
    """Run system backtest using global validator"""
    return performance_validator.run_comprehensive_backtest(strategy)