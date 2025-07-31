#!/usr/bin/env python3
"""
Enhanced DreamTeamAI - Setup and Run Script
Handles dependency installation and graceful startup
"""

import sys
import subprocess
import importlib.util
import os

def check_and_install_dependency(package_name, install_name=None):
    """Check if a package is installed, install if missing"""
    if install_name is None:
        install_name = package_name
    
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        print(f"📦 Installing missing dependency: {install_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {install_name}")
            return False

def setup_dependencies():
    """Setup essential dependencies"""
    print("🔧 Checking and installing dependencies...")
    
    essential_deps = [
        ("numpy", "numpy>=1.20.0"),
        ("pandas", "pandas>=1.3.0"),
        ("requests", "requests>=2.25.0"),
        ("aiohttp", "aiohttp>=3.8.0"),
        ("sklearn", "scikit-learn>=1.0.0"),
        ("scipy", "scipy>=1.7.0"),
        ("dateutil", "python-dateutil>=2.8.0"),
    ]
    
    optional_deps = [
        ("ortools", "ortools>=9.4.0"),
        ("xgboost", "xgboost>=1.5.0"),
        ("tqdm", "tqdm>=4.60.0"),
    ]
    
    # Check essential dependencies
    missing_essential = []
    for package, install_name in essential_deps:
        if not check_and_install_dependency(package, install_name):
            missing_essential.append(package)
    
    if missing_essential:
        print(f"❌ Critical dependencies missing: {missing_essential}")
        print("Please install manually: pip install -r requirements_minimal.txt")
        return False
    
    # Check optional dependencies
    for package, install_name in optional_deps:
        check_and_install_dependency(package, install_name)
    
    print("✅ Dependencies setup complete!")
    return True

def run_enhanced_system(query=None):
    """Run the enhanced DreamTeamAI system"""
    if query is None:
        query = "india vs australia"  # Default query
    
    print(f"🚀 Starting Enhanced DreamTeamAI with query: {query}")
    
    try:
        # Import and run the enhanced system
        from enhanced_dreamteam_ai import EnhancedDreamTeamAI
        import asyncio
        
        async def main():
            enhanced_ai = EnhancedDreamTeamAI()
            results = await enhanced_ai.generate_enhanced_teams(query, num_teams=3)
            return results
        
        # Run the system
        results = asyncio.run(main())
        
        if results.get('success'):
            print("🎉 Team generation completed successfully!")
            teams = results.get('teams', [])
            print(f"📊 Generated {len(teams)} optimized teams")
            
            # Display top team
            if teams:
                top_team = teams[0]
                print(f"\n🏆 TOP TEAM (Score: {top_team.get('total_score', 0):.1f}):")
                players = top_team.get('players', [])
                for i, player in enumerate(players[:11], 1):
                    print(f"  {i}. {player.get('name', 'Unknown')} - {player.get('credits', 0)} credits")
        else:
            print("⚠️  Team generation encountered issues, but system is working")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Falling back to simplified mode...")
        run_simplified_mode(query)
    except Exception as e:
        print(f"❌ Error running enhanced system: {e}")
        print("🔄 Falling back to simplified mode...")
        run_simplified_mode(query)

def run_simplified_mode(query):
    """Run simplified mode without advanced AI features"""
    print("🔄 Running in simplified mode...")
    try:
        # Import basic system
        from run_dreamteam import main as run_basic
        run_basic()
    except Exception as e:
        print(f"❌ Error in simplified mode: {e}")
        print("\n📖 MANUAL SETUP REQUIRED:")
        print("1. pip install -r requirements_minimal.txt")
        print("2. python3 enhanced_dreamteam_ai.py 'your match query'")

def main():
    """Main setup and run function"""
    print("🏆 Enhanced DreamTeamAI - Setup and Run")
    print("=" * 50)
    
    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "india vs australia"
    
    # Setup dependencies
    if setup_dependencies():
        # Run the enhanced system
        run_enhanced_system(query)
    else:
        print("❌ Setup failed. Please install dependencies manually.")
        print("Run: pip install -r requirements_minimal.txt")

if __name__ == "__main__":
    main()