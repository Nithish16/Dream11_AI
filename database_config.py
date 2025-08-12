#!/usr/bin/env python3
"""
🗄️ Database Configuration Access
Pure database-driven configuration system
"""

import sqlite3
import json
import os
from typing import Any, List, Optional

class DatabaseConfig:
    """
    🗄️ Access system configuration from database
    """
    
    def __init__(self, db_path: str = 'universal_cricket_intelligence.db'):
        self.db_path = db_path
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT config_value, config_type FROM system_config WHERE config_key = ?',
                (key,)
            )
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                value, config_type = result
                
                # Parse based on type
                if config_type == 'json_list':
                    return json.loads(value)
                elif config_type == 'json_object':
                    return json.loads(value)
                else:
                    return value
            
            return default
            
        except Exception as e:
            print(f"⚠️ Error reading config {key}: {e}")
            return default
    
    def set_config(self, key: str, value: Any, description: str = None) -> bool:
        """Set configuration value in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine config type and serialize value
            if isinstance(value, (list, dict)):
                config_value = json.dumps(value)
                config_type = 'json_list' if isinstance(value, list) else 'json_object'
            else:
                config_value = str(value)
                config_type = 'string'
            
            cursor.execute('''
                INSERT OR REPLACE INTO system_config 
                (config_key, config_value, config_type, description, updated_date)
                VALUES (?, ?, ?, ?, datetime('now'))
            ''', (key, config_value, config_type, description))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error setting config {key}: {e}")
            return False
    
    def get_skip_series(self) -> List[str]:
        """Get list of series to skip"""
        return self.get_config('skip_series_list', [])
    
    def add_skip_series(self, series_name: str) -> bool:
        """Add series to skip list"""
        skip_list = self.get_skip_series()
        if series_name not in skip_list:
            skip_list.append(series_name)
            return self.set_config('skip_series_list', skip_list, 'List of series to skip')
        return True
    
    def remove_skip_series(self, series_name: str) -> bool:
        """Remove series from skip list"""
        skip_list = self.get_skip_series()
        if series_name in skip_list:
            skip_list.remove(series_name)
            return self.set_config('skip_series_list', skip_list, 'List of series to skip')
        return True

# Convenient global instance
db_config = DatabaseConfig()

if __name__ == '__main__':
    # CLI interface for config management
    import sys
    
    if len(sys.argv) < 2:
        print('🗄️ Database Configuration Tool')
        print('Usage:')
        print('  python3 database_config.py get <key>')
        print('  python3 database_config.py set <key> <value>')
        print('  python3 database_config.py skip-list')
        print('  python3 database_config.py skip-add <series>')
        print('  python3 database_config.py skip-remove <series>')
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == 'get' and len(sys.argv) == 3:
        value = db_config.get_config(sys.argv[2])
        print(f'{sys.argv[2]}: {value}')
    
    elif action == 'set' and len(sys.argv) == 4:
        success = db_config.set_config(sys.argv[2], sys.argv[3])
        print('✅ Config updated' if success else '❌ Config update failed')
    
    elif action == 'skip-list':
        skip_list = db_config.get_skip_series()
        print(f'Skip series: {skip_list}')
    
    elif action == 'skip-add' and len(sys.argv) == 3:
        success = db_config.add_skip_series(sys.argv[2])
        print('✅ Series added to skip list' if success else '❌ Failed to add series')
    
    elif action == 'skip-remove' and len(sys.argv) == 3:
        success = db_config.remove_skip_series(sys.argv[2])
        print('✅ Series removed from skip list' if success else '❌ Failed to remove series')
    
    else:
        print('❌ Invalid command')
        sys.exit(1)
