#!/usr/bin/env python3
"""
Dream11 AI - Intelligent Dependency Installer
Automatically installs the right dependencies for your system with intelligent fallbacks
"""

import subprocess
import sys
import platform
import importlib
import time
from typing import List, Dict, Tuple, Optional

def print_banner():
    """Print installation banner"""
    print("🚀 Dream11 AI - Intelligent Dependency Installer")
    print("=" * 60)
    print("🔍 Analyzing system capabilities...")
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {platform.platform()}")
    print(f"🏗️ Architecture: {platform.machine()}")
    print()

def check_package(package: str) -> Tuple[bool, Optional[str]]:
    """Check if a package is installed and return version if available"""
    try:
        module = importlib.import_module(package.replace('-', '_'))
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def install_package(package: str, description: str = "") -> bool:
    """Install a package using pip"""
    try:
        print(f"📦 Installing {package}{'(' + description + ')' if description else ''}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
            return True
        else:
            print(f"❌ Failed to install {package}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Installation of {package} timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")
        return False

def install_with_fallback(packages: List[str], description: str = "") -> bool:
    """Try to install packages with fallbacks"""
    for package in packages:
        if install_package(package, description):
            return True
    return False

def main():
    """Main installation process"""
    print_banner()
    
    # Track installation results
    installed = []
    failed = []
    skipped = []
    
    # Define package categories with fallbacks
    package_categories = {
        "Core Dependencies (Required)": {
            "packages": [
                ("requests", "HTTP client for API calls"),
                ("python-dateutil", "Date/time utilities"),
                ("urllib3", "HTTP library"),
            ],
            "required": True
        },
        "Machine Learning (Enhanced AI)": {
            "packages": [
                ("numpy", "Numerical computing"),
                ("pandas", "Data manipulation"),
                ("scikit-learn", "Machine learning"),
            ],
            "required": False,
            "fallback_message": "Will use statistical fallbacks"
        },
        "Advanced AI (Optional)": {
            "packages": [
                ("torch", "Neural networks"),
                ("scipy", "Scientific computing"),
            ],
            "required": False,
            "fallback_message": "Advanced features will use fallbacks"
        },
        "Optimization (Optional)": {
            "packages": [
                ("cvxpy", "Convex optimization"),
                ("ortools", "Google OR-Tools"),
            ],
            "required": False,
            "fallback_message": "Will use greedy optimization"
        },
        "Performance (Optional)": {
            "packages": [
                ("aiohttp", "Async HTTP"),
                ("joblib", "Parallel processing"),
            ],
            "required": False,
            "fallback_message": "Performance optimizations disabled"
        },
        "Visualization (Optional)": {
            "packages": [
                ("matplotlib", "Plotting"),
                ("seaborn", "Statistical visualization"),
            ],
            "required": False,
            "fallback_message": "Text-only output"
        }
    }
    
    print("🔍 Checking existing packages...")
    time.sleep(1)
    
    # Process each category
    for category_name, category_info in package_categories.items():
        print(f"\n📋 {category_name}")
        print("-" * 40)
        
        category_success = True
        category_installed = []
        category_failed = []
        
        for package, description in category_info["packages"]:
            # Check if already installed
            is_installed, version = check_package(package)
            
            if is_installed:
                print(f"✅ {package} ({version}) - Already installed")
                category_installed.append(package)
                skipped.append(package)
            else:
                # Try to install
                success = install_package(package, description)
                if success:
                    category_installed.append(package)
                    installed.append(package)
                else:
                    category_failed.append(package)
                    failed.append(package)
                    if category_info["required"]:
                        category_success = False
                
                # Brief pause between installations
                time.sleep(0.5)
        
        # Category summary
        if category_installed:
            print(f"✅ Installed in this category: {', '.join(category_installed)}")
        
        if category_failed:
            print(f"❌ Failed in this category: {', '.join(category_failed)}")
            if not category_info["required"] and "fallback_message" in category_info:
                print(f"🔄 Fallback: {category_info['fallback_message']}")
        
        if category_info["required"] and not category_success:
            print(f"🚨 Critical: Some required packages failed to install!")
            print(f"💡 Try: pip install --upgrade pip")
            print(f"💡 Or: conda install {' '.join([p[0] for p in category_info['packages']])}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    
    print(f"✅ Successfully installed: {len(installed)}")
    if installed:
        for package in installed:
            print(f"    • {package}")
    
    print(f"⏭️ Already installed: {len(skipped)}")
    if skipped:
        for package in skipped:
            print(f"    • {package}")
    
    print(f"❌ Failed to install: {len(failed)}")
    if failed:
        for package in failed:
            print(f"    • {package}")
    
    # System readiness assessment
    core_packages = ["requests", "python-dateutil", "urllib3"]
    core_installed = all(check_package(pkg)[0] for pkg in core_packages)
    
    ml_packages = ["numpy", "pandas", "scikit-learn"]
    ml_installed = any(check_package(pkg)[0] for pkg in ml_packages)
    
    advanced_packages = ["torch", "cvxpy"]
    advanced_installed = any(check_package(pkg)[0] for pkg in advanced_packages)
    
    print(f"\n🎯 SYSTEM READINESS:")
    print(f"    {'✅' if core_installed else '❌'} Core functionality: {'Ready' if core_installed else 'Missing required packages'}")
    print(f"    {'✅' if ml_installed else '🔄'} Enhanced AI: {'Available' if ml_installed else 'Fallback mode'}")
    print(f"    {'✅' if advanced_installed else '🔄'} Advanced features: {'Available' if advanced_installed else 'Fallback mode'}")
    
    if core_installed:
        print(f"\n🚀 Dream11 AI is ready to run!")
        print(f"💻 Try: python3 dream11_ai.py 105780 5")
    else:
        print(f"\n⚠️ Please install missing core dependencies before running")
        print(f"💡 Manual install: pip install requests python-dateutil urllib3")
    
    # Platform-specific recommendations
    if platform.machine().lower() in ['arm64', 'aarch64']:
        print(f"\n🍎 Apple Silicon detected:")
        print(f"💡 For better ML performance, consider using conda:")
        print(f"   conda install numpy pandas scikit-learn")
    
    if sys.version_info >= (3, 12):
        print(f"\n🐍 Python 3.12+ detected:")
        print(f"💡 Some packages may need newer versions")
        print(f"💡 If issues persist, consider Python 3.11 or conda environment")
    
    print("\n" + "=" * 60)
    return 0 if core_installed else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Installation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)