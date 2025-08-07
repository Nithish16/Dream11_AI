#!/usr/bin/env python3
"""
Quick System Health Check for Dream11 AI Production System
Run this to verify the system is ready for use
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        item_count = len(os.listdir(dirpath))
        print(f"✅ {description}: {dirpath} ({item_count} items)")
        return True
    else:
        print(f"❌ {description}: {dirpath} - NOT FOUND")
        return False

def check_python_import(module_name):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ Module: {module_name}")
        return True
    except ImportError as e:
        print(f"⚠️ Module: {module_name} - {e}")
        return False

def run_quick_test():
    """Run a quick system test"""
    try:
        print("🧪 Running quick system test...")
        result = subprocess.run([
            sys.executable, 'final_validation_test.py'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and "100.0%" in result.stdout:
            print("✅ System test: PASSED")
            return True
        else:
            print("⚠️ System test: Issues detected")
            return False
    except Exception as e:
        print(f"❌ System test: Failed - {e}")
        return False

def main():
    """Run comprehensive health check"""
    print("🏆 DREAM11 AI - SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 0
    
    # Check core files
    print("\n📁 Core Files:")
    core_files = [
        ('dream11_ai.py', 'Main entry point'),
        ('install_dependencies.py', 'Dependency installer'),
        ('production_test_suite.py', 'Production testing'),
        ('final_validation_test.py', 'System validation'),
        ('dependency_manager.py', 'Dependency manager'),
        ('requirements.txt', 'Requirements'),
        ('README.md', 'Documentation')
    ]
    
    for filepath, desc in core_files:
        if check_file_exists(filepath, desc):
            checks_passed += 1
        total_checks += 1
    
    # Check directories
    print("\n📂 Directories:")
    directories = [
        ('core_logic', 'AI modules'),
        ('utils', 'Utilities'),
        ('tests', 'Test suites'),
        ('archive', 'Archived files')
    ]
    
    for dirpath, desc in directories:
        if check_directory_exists(dirpath, desc):
            checks_passed += 1
        total_checks += 1
    
    # Check core modules
    print("\n🧠 Core Modules:")
    core_modules = [
        'core_logic.team_generator',
        'core_logic.data_aggregator',
        'core_logic.feature_engine',
        'core_logic.match_resolver'
    ]
    
    for module in core_modules:
        if check_python_import(module):
            checks_passed += 1
        total_checks += 1
    
    # Check main system
    print("\n🎯 Main System:")
    try:
        from dream11_ai import Dream11ProductionAI
        print("✅ Main system: Import successful")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Main system: Import failed - {e}")
    total_checks += 1
    
    # Run quick test
    print("\n🧪 Quick Test:")
    if run_quick_test():
        checks_passed += 1
    total_checks += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"🧪 Checks run: {total_checks}")
    print(f"✅ Checks passed: {checks_passed}")
    print(f"❌ Checks failed: {total_checks - checks_passed}")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 SYSTEM STATUS: ✅ HEALTHY")
        print("🚀 Ready for production use!")
        return True
    elif success_rate >= 70:
        print("\n⚠️ SYSTEM STATUS: 🔶 NEEDS ATTENTION")
        print("🔧 Some issues need to be addressed")
        return False
    else:
        print("\n❌ SYSTEM STATUS: 🔴 CRITICAL ISSUES")
        print("🛠️ System requires maintenance")
        return False

if __name__ == "__main__":
    try:
        healthy = main()
        sys.exit(0 if healthy else 1)
    except KeyboardInterrupt:
        print("\n🛑 Health check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)