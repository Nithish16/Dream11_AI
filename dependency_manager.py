#!/usr/bin/env python3
"""
Comprehensive Dependency Manager for Dream11 AI
Handles all ML dependencies with fallbacks and compatibility checks
"""

import sys
import subprocess
import importlib
import platform
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages all dependencies with intelligent fallbacks"""

    def __init__(self):
        self.python_version = sys.version_info
        self.platform_info = platform.platform()
        self.architecture = platform.machine()
        self.is_arm64 = self.architecture.lower() in ['arm64', 'aarch64']

        # Dependency categories with fallbacks
        self.core_deps = {
            'requests': {'required': True, 'fallback': None},
            'python-dateutil': {'required': True, 'fallback': None},
            'urllib3': {'required': True, 'fallback': None}
        }

        self.ml_deps = {
            'numpy': {'required': True, 'fallback': 'manual_numpy'},
            'pandas': {'required': True, 'fallback': 'csv_processing'},
            'scipy': {'required': False, 'fallback': 'basic_stats'},
            'scikit-learn': {'required': False, 'fallback': 'manual_ml'},
            'tensorflow': {'required': False, 'fallback': 'pytorch'},
            'torch': {'required': False, 'fallback': 'manual_neural'},
            'transformers': {'required': False, 'fallback': 'basic_nlp'}
        }

        self.optimization_deps = {
            'cvxpy': {'required': False, 'fallback': 'pulp'},
            'pulp': {'required': False, 'fallback': 'basic_optimization'},
            'ortools': {'required': False, 'fallback': 'manual_optimization'}
        }

        self.viz_deps = {
            'matplotlib': {'required': False, 'fallback': 'text_output'},
            'seaborn': {'required': False, 'fallback': 'matplotlib'},
            'plotly': {'required': False, 'fallback': 'basic_charts'}
        }

        self.performance_deps = {
            'numba': {'required': False, 'fallback': 'pure_python'},
            'joblib': {'required': False, 'fallback': 'threading'},
            'dask': {'required': False, 'fallback': 'sequential'},
            'aiohttp': {'required': False, 'fallback': 'requests'}
        }

        self.available_deps = {}
        self.fallback_implementations = {}

    def check_all_dependencies(self) -> Dict[str, Any]:
        """Check all dependencies and return status"""
        logger.info("🔍 Checking all dependencies...")

        status = {
            'core': self._check_dependency_group(self.core_deps),
            'ml': self._check_dependency_group(self.ml_deps),
            'optimization': self._check_dependency_group(self.optimization_deps),
            'visualization': self._check_dependency_group(self.viz_deps),
            'performance': self._check_dependency_group(self.performance_deps),
            'system_info': {
                'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                'platform': self.platform_info,
                'architecture': self.architecture,
                'is_arm64': self.is_arm64
            }
        }

        self._initialize_fallbacks()
        return status

    def _check_dependency_group(self, deps: Dict) -> Dict[str, Dict]:
        """Check a group of dependencies"""
        results = {}

        for dep_name, config in deps.items():
            try:
                # Try to import the module
                module = importlib.import_module(dep_name.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')

                results[dep_name] = {
                    'available': True,
                    'version': version,
                    'required': config['required'],
                    'fallback': config['fallback']
                }
                self.available_deps[dep_name] = module

            except ImportError as e:
                results[dep_name] = {
                    'available': False,
                    'error': str(e),
                    'required': config['required'],
                    'fallback': config['fallback']
                }

        return results

    def _initialize_fallbacks(self):
        """Initialize fallback implementations for missing dependencies"""

        # NumPy fallback
        if 'numpy' not in self.available_deps:
            self.fallback_implementations['numpy'] = self._create_numpy_fallback()

        # Pandas fallback
        if 'pandas' not in self.available_deps:
            self.fallback_implementations['pandas'] = self._create_pandas_fallback()

        # ML fallbacks
        if 'scikit-learn' not in self.available_deps:
            self.fallback_implementations['sklearn'] = self._create_sklearn_fallback()

        # Optimization fallbacks
        if 'cvxpy' not in self.available_deps and 'pulp' not in self.available_deps:
            self.fallback_implementations['optimization'] = self._create_optimization_fallback()

    def _create_numpy_fallback(self):
        """Create numpy-like functionality using pure Python"""
        class NumpyFallback:
            @staticmethod
            def array(data):
                return list(data) if not isinstance(data, list) else data

            @staticmethod
            def mean(data):
                return sum(data) / len(data) if data else 0

            @staticmethod
            def std(data):
                if not data:
                    return 0
                mean_val = sum(data) / len(data)
                variance = sum((x - mean_val) ** 2 for x in data) / len(data)
                return variance ** 0.5

            @staticmethod
            def random_normal(mean=0, std=1, size=1):
                import random
                return [random.gauss(mean, std) for _ in range(size)]

            @staticmethod
            def linspace(start, stop, num):
                step = (stop - start) / (num - 1)
                return [start + i * step for i in range(num)]

        return NumpyFallback()

    def _create_pandas_fallback(self):
        """Create pandas-like functionality using pure Python"""
        class PandasFallback:
            class DataFrame:
                def __init__(self, data=None):
                    self.data = data or {}

                def to_dict(self, orient='dict'):
                    return self.data

                def mean(self):
                    result = {}
                    for col, values in self.data.items():
                        if isinstance(values[0], (int, float)):
                            result[col] = sum(values) / len(values)
                    return result

            @staticmethod
            def read_csv(filepath):
                # Basic CSV reading
                import csv
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                return PandasFallback.DataFrame(data)

        return PandasFallback()

    def _create_sklearn_fallback(self):
        """Create basic ML functionality"""
        class SklearnFallback:
            class LinearRegression:
                def __init__(self):
                    self.coef_ = None
                    self.intercept_ = None

                def fit(self, X, y):
                    # Simple linear regression using least squares
                    n = len(X)
                    sum_x = sum(X)
                    sum_y = sum(y)
                    sum_xy = sum(x * y for x, y in zip(X, y))
                    sum_x2 = sum(x * x for x in X)

                    self.coef_ = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    self.intercept_ = (sum_y - self.coef_ * sum_x) / n
                    return self

                def predict(self, X):
                    return [self.coef_ * x + self.intercept_ for x in X]

            class StandardScaler:
                def __init__(self):
                    self.mean_ = None
                    self.scale_ = None

                def fit(self, X):
                    self.mean_ = [sum(col) / len(col) for col in zip(*X)]
                    variances = []
                    for i, mean_val in enumerate(self.mean_):
                        variance = sum((row[i] - mean_val) ** 2 for row in X) / len(X)
                        variances.append(variance ** 0.5)
                    self.scale_ = variances
                    return self

                def transform(self, X):
                    return [[(row[i] - self.mean_[i]) / self.scale_[i]
                            for i in range(len(row))] for row in X]

        return SklearnFallback()

    def _create_optimization_fallback(self):
        """Create basic optimization functionality"""
        class OptimizationFallback:
            @staticmethod
            def maximize_linear(coefficients, constraints, bounds):
                """Simple greedy optimization"""
                # Sort by coefficient value (highest first)
                items = list(enumerate(coefficients))
                items.sort(key=lambda x: x[1], reverse=True)

                result = [0] * len(coefficients)
                total_cost = 0

                for idx, coeff in items:
                    lower, upper = bounds[idx]
                    max_allowed = min(upper, constraints.get('budget', float('inf')) - total_cost)

                    if max_allowed > lower:
                        result[idx] = max_allowed
                        total_cost += max_allowed

                return result

        return OptimizationFallback()

    def get_safe_import(self, module_name: str) -> Any:
        """Get module with fallback if available"""
        if module_name in self.available_deps:
            return self.available_deps[module_name]
        elif module_name in self.fallback_implementations:
            logger.warning(f"⚠️ Using fallback implementation for {module_name}")
            return self.fallback_implementations[module_name]
        else:
            logger.error(f"❌ {module_name} not available and no fallback implemented")
            return None

    def install_compatible_dependencies(self) -> bool:
        """Install compatible versions of dependencies"""
        logger.info("📦 Installing compatible dependencies...")

        success = True

        # Core dependencies (always needed)
        core_packages = ['requests', 'python-dateutil']
        for package in core_packages:
            if not self._install_package(package):
                success = False

        # ML dependencies with version compatibility
        if self.python_version >= (3, 9) and self.python_version < (3, 12):
            ml_packages = [
                'numpy>=1.21.0,<2.0.0',
                'pandas>=1.3.0,<3.0.0',
                'scikit-learn>=1.0.0,<2.0.0'
            ]

            for package in ml_packages:
                self._install_package(package, ignore_errors=True)

        # PyTorch for supported versions
        if self.python_version < (3, 12):
            torch_cmd = self._get_torch_install_command()
            if torch_cmd:
                self._run_pip_command(torch_cmd, ignore_errors=True)

        return success

    def _install_package(self, package: str, ignore_errors: bool = False) -> bool:
        """Install a single package"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info(f"✅ Successfully installed {package}")
                return True
            else:
                if not ignore_errors:
                    logger.error(f"❌ Failed to install {package}: {result.stderr}")
                return False

        except Exception as e:
            if not ignore_errors:
                logger.error(f"❌ Error installing {package}: {e}")
            return False

    def _get_torch_install_command(self) -> Optional[List[str]]:
        """Get appropriate PyTorch install command"""
        if self.is_arm64 and sys.platform == 'darwin':  # Apple Silicon Mac
            return [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu']
        elif self.python_version < (3, 11):
            return [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision']
        else:
            return None

    def _run_pip_command(self, cmd: List[str], ignore_errors: bool = False) -> bool:
        """Run pip command safely"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"✅ Command successful: {' '.join(cmd)}")
                return True
            else:
                if not ignore_errors:
                    logger.error(f"❌ Command failed: {result.stderr}")
                return False
        except Exception as e:
            if not ignore_errors:
                logger.error(f"❌ Command error: {e}")
            return False

    def create_minimal_environment(self) -> Dict[str, Any]:
        """Create minimal working environment"""
        logger.info("🛠️ Creating minimal working environment...")

        # Install only essential packages
        essential = ['requests']
        for package in essential:
            self._install_package(package)

        # Check what's available after installation
        status = self.check_all_dependencies()

        return {
            'status': 'minimal',
            'available_features': self._get_available_features(status),
            'limitations': self._get_limitations(status),
            'recommendations': self._get_recommendations(status)
        }

    def _get_available_features(self, status: Dict) -> List[str]:
        """Get list of available features"""
        features = ['Basic team generation', 'Match resolution', 'Data aggregation']

        if status['ml']['numpy']['available']:
            features.append('Advanced statistical analysis')

        if status['ml']['pandas']['available']:
            features.append('Data processing and analysis')

        if status['ml']['scikit-learn']['available']:
            features.append('Machine learning predictions')

        if status['optimization']['pulp']['available'] or status['optimization']['cvxpy']['available']:
            features.append('Mathematical optimization')

        return features

    def _get_limitations(self, status: Dict) -> List[str]:
        """Get list of current limitations"""
        limitations = []

        if not status['ml']['numpy']['available']:
            limitations.append('Limited numerical computations')

        if not status['ml']['torch']['available'] and not status['ml']['tensorflow']['available']:
            limitations.append('No deep learning capabilities')

        if not status['optimization']['cvxpy']['available']:
            limitations.append('Basic optimization only')

        return limitations

    def _get_recommendations(self, status: Dict) -> List[str]:
        """Get recommendations for improvement"""
        recommendations = []

        if self.python_version >= (3, 12):
            recommendations.append('Consider using Python 3.11 or earlier for better ML library support')

        if self.is_arm64:
            recommendations.append('Some ML libraries may have limited ARM64 support')

        if not status['ml']['numpy']['available']:
            recommendations.append('Try: pip install numpy --only-binary=:all:')

        return recommendations

# Global dependency manager instance
dep_manager = DependencyManager()

def get_dependency_manager() -> DependencyManager:
    """Get the global dependency manager instance"""
    return dep_manager

def verify_dependencies():
    """
    Quick dependency verification for imports
    Returns True if all critical dependencies are available
    """
    try:
        manager = DependencyManager()
        deps = manager.check_core_dependencies()
        
        # Check critical imports
        critical_deps = ['requests', 'python-dateutil']
        all_good = True
        
        for dep in critical_deps:
            if dep not in deps or not deps[dep]['available']:
                all_good = False
                break
        
        return all_good
    except:
        return False
