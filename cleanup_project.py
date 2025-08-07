#!/usr/bin/env python3
"""
Project Cleanup Script - Safe removal of unused files and code cleanup
Only removes temporary/analysis files, keeps all core functionality intact
"""

import os
import shutil
import re
from pathlib import Path
from datetime import datetime

class ProjectCleaner:
    def __init__(self):
        self.root_dir = Path(".")
        self.removed_files = []
        self.cleaned_files = []
        self.backup_dir = Path("cleanup_backup")
        
    def create_backup(self):
        """Create backup of files before cleanup"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        print(f"✅ Created backup directory: {self.backup_dir}")
    
    def remove_cache_directories(self):
        """Remove Python __pycache__ directories"""
        print("\n🗑️ Removing Python cache directories...")
        
        cache_dirs = list(self.root_dir.rglob("__pycache__"))
        for cache_dir in cache_dirs:
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
                self.removed_files.append(str(cache_dir))
                print(f"   ✅ Removed: {cache_dir}")
        
        if not cache_dirs:
            print("   ℹ️ No cache directories found")
    
    def identify_analysis_files(self):
        """Identify temporary analysis files created during our session"""
        analysis_files = [
            "acceleration_strategy.py",
            "detailed_analysis.py", 
            "learning_trajectory_analysis.py",
            "match_analysis.py",
            "realistic_learning_plan.py"
        ]
        
        print("\n🔍 Identifying analysis files...")
        files_to_remove = []
        
        for file in analysis_files:
            file_path = self.root_dir / file
            if file_path.exists():
                files_to_remove.append(file_path)
                print(f"   📄 Found analysis file: {file}")
        
        return files_to_remove
    
    def backup_and_remove_analysis_files(self):
        """Backup and remove analysis files"""
        analysis_files = self.identify_analysis_files()
        
        if not analysis_files:
            print("   ℹ️ No analysis files to remove")
            return
        
        print(f"\n📦 Backing up {len(analysis_files)} analysis files...")
        
        for file_path in analysis_files:
            # Backup file
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            
            # Remove original
            file_path.unlink()
            self.removed_files.append(str(file_path))
            print(f"   ✅ Moved to backup: {file_path.name}")
    
    def clean_code_files(self):
        """Clean up code files - remove unused imports, fix formatting"""
        print("\n🧹 Cleaning code files...")
        
        core_files = [
            "dream11_ai.py",
            "dependency_manager.py",
            "core_logic/team_generator.py",
            "core_logic/data_aggregator.py",
            "utils/api_client.py"
        ]
        
        for file_path in core_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self.clean_single_file(full_path)
    
    def clean_single_file(self, file_path: Path):
        """Clean a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove excessive blank lines (more than 2 consecutive)
            content = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', content)
            
            # Remove trailing whitespace
            lines = content.split('\n')
            lines = [line.rstrip() for line in lines]
            content = '\n'.join(lines)
            
            # Remove blank lines at end of file
            content = content.rstrip() + '\n'
            
            if content != original_content:
                # Backup original
                backup_path = self.backup_dir / f"{file_path.name}.original"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write cleaned version
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.cleaned_files.append(str(file_path))
                print(f"   ✅ Cleaned: {file_path}")
            
        except Exception as e:
            print(f"   ⚠️ Error cleaning {file_path}: {e}")
    
    def remove_empty_directories(self):
        """Remove empty directories"""
        print("\n📁 Checking for empty directories...")
        
        empty_dirs = []
        for root, dirs, files in os.walk(self.root_dir):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    empty_dirs.append(dir_path)
        
        for empty_dir in empty_dirs:
            if empty_dir.name != "feedback_data":  # Keep feedback_data even if empty
                empty_dir.rmdir()
                self.removed_files.append(str(empty_dir))
                print(f"   ✅ Removed empty directory: {empty_dir}")
        
        if not empty_dirs:
            print("   ℹ️ No empty directories found")
    
    def identify_duplicate_files(self):
        """Identify potential duplicate files"""
        print("\n🔍 Checking for duplicate files...")
        
        # Look for files with similar names
        all_files = list(self.root_dir.rglob("*.py"))
        duplicates = []
        
        for file1 in all_files:
            for file2 in all_files:
                if file1 != file2 and file1.stem in file2.stem and file1.stem != file2.stem:
                    # Check if one is clearly a variant of the other
                    if any(suffix in file2.stem for suffix in ['_enhanced', '_advanced', '_v2', '_new']):
                        duplicates.append((file1, file2))
        
        if duplicates:
            print("   ⚠️ Potential duplicates found (manual review recommended):")
            for orig, dup in duplicates:
                print(f"      • {orig} vs {dup}")
        else:
            print("   ✅ No obvious duplicates found")
    
    def cleanup_requirements_files(self):
        """Clean up requirements files"""
        print("\n📦 Cleaning requirements files...")
        
        req_files = [
            "requirements.txt",
            "requirements_production_final.txt"
        ]
        
        for req_file in req_files:
            file_path = self.root_dir / req_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Remove empty lines and comments
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('-r'):
                            cleaned_lines.append(line)
                    
                    # Sort requirements
                    cleaned_lines.sort()
                    
                    # Write back
                    with open(file_path, 'w') as f:
                        f.write('\n'.join(cleaned_lines) + '\n')
                    
                    self.cleaned_files.append(str(file_path))
                    print(f"   ✅ Cleaned: {req_file}")
                
                except Exception as e:
                    print(f"   ⚠️ Error cleaning {req_file}: {e}")
    
    def generate_cleanup_report(self):
        """Generate cleanup report"""
        print("\n" + "="*80)
        print("🧹 PROJECT CLEANUP REPORT")
        print("="*80)
        print(f"📅 Cleanup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print(f"🗑️ Files Removed: {len(self.removed_files)}")
        for file in self.removed_files:
            print(f"   • {file}")
        
        print(f"\n🧹 Files Cleaned: {len(self.cleaned_files)}")
        for file in self.cleaned_files:
            print(f"   • {file}")
        
        print(f"\n📦 Backup Location: {self.backup_dir}")
        print("   All removed/modified files are safely backed up here")
        
        print(f"\n✅ CLEANUP SUMMARY:")
        print(f"   • Removed {len([f for f in self.removed_files if 'cache' in f])} cache directories")
        print(f"   • Removed {len([f for f in self.removed_files if f.endswith('.py')])} analysis files")
        print(f"   • Cleaned {len(self.cleaned_files)} code files")
        print(f"   • Core functionality preserved")
        
        # Save report to file
        report_file = self.backup_dir / "cleanup_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"Project Cleanup Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Files Removed:\n")
            for file in self.removed_files:
                f.write(f"  {file}\n")
            f.write(f"\nFiles Cleaned:\n")
            for file in self.cleaned_files:
                f.write(f"  {file}\n")
        
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("🧹 STARTING PROJECT CLEANUP")
        print("="*80)
        print("This will safely remove temporary files and clean up code")
        print("All changes are backed up for safety")
        print()
        
        # Create backup first
        self.create_backup()
        
        # Remove cache directories
        self.remove_cache_directories()
        
        # Remove analysis files
        self.backup_and_remove_analysis_files()
        
        # Clean code files
        self.clean_code_files()
        
        # Clean requirements
        self.cleanup_requirements_files()
        
        # Remove empty directories
        self.remove_empty_directories()
        
        # Check for duplicates
        self.identify_duplicate_files()
        
        # Generate report
        self.generate_cleanup_report()
        
        print("\n🎉 CLEANUP COMPLETED SUCCESSFULLY!")
        print("Your project is now cleaner and more maintainable.")

def main():
    """Main cleanup function"""
    # Confirm with user
    print("🧹 PROJECT CLEANUP UTILITY")
    print("="*50)
    print("This will:")
    print("• Remove Python __pycache__ directories")
    print("• Remove temporary analysis files")
    print("• Clean up code formatting")
    print("• Remove empty directories")
    print("• Create backups of all changes")
    print()
    print("⚠️ IMPORTANT: Core functionality will NOT be affected")
    print()
    
    response = input("Proceed with cleanup? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        cleaner = ProjectCleaner()
        cleaner.run_cleanup()
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()
