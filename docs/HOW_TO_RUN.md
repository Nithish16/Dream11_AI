# 🚀 DreamTeamAI - How to Run Guide

## 📋 **Prerequisites**

### **System Requirements**
- **Python 3.7+** (Tested with Python 3.13.5)
- **macOS/Linux/Windows** compatible
- **Internet connection** for API calls
- **8GB RAM** recommended (4GB minimum)

### **Required Dependencies**
The following packages are needed (automatically installed):
```
pandas>=1.5.0
numpy>=1.23.0
requests>=2.28.0
ortools>=9.4.0
scikit-learn>=1.1.0
xgboost>=1.6.0
flask>=2.2.0
fastapi>=0.95.0
uvicorn>=0.20.0
python-dateutil>=2.8.0
```

## 🔧 **Installation Steps**

### **Step 1: Download/Clone the Project**
```bash
# If you have the project files
cd /path/to/Dream11_AI

# Or clone from repository (if available)
git clone <repository-url>
cd Dream11_AI
```

### **Step 2: Install Dependencies**
```bash
# Install required packages
pip3 install -r requirements.txt

# Or install manually
pip3 install pandas numpy requests ortools scikit-learn xgboost flask fastapi uvicorn python-dateutil
```

### **Step 3: Verify Installation**
```bash
# Test core imports
python3 -c "
from core_logic.match_resolver import resolve_match_by_id
from core_logic.team_generator import generate_hybrid_teams
from ortools.linear_solver import pywraplp
print('✅ All dependencies installed successfully!')
"
```

## 🏃‍♂️ **Running the Application**

### **Method 1: Interactive Menu (Recommended)**
```bash
# Navigate to project directory
cd /Users/nitish.natarajan/Downloads/Dream11_AI

# Run the main application
python3 run_dreamteam.py
```

**What you'll see:**
```
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆
🚀 WELCOME TO DREAMTEAMAI - DREAM11 PREDICTOR 🚀
🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆

📋 CHOOSE YOUR OPTION:
========================================
1️⃣  🎯 Generate Hybrid Dream11 Teams (Match ID)
2️⃣  📊 Quick Team Preview
3️⃣  ❓ Help & Info
4️⃣  🚪 Exit
========================================
```

### **Method 2: Direct Testing**
```bash
# Test with sample workflow
python3 test_hybrid_workflow.py

# Test complete workflow
python3 test_complete_workflow.py
```

## 🎯 **Using the Application**

### **Step-by-Step User Guide**

1. **Start the Application**
   ```bash
   python3 run_dreamteam.py
   ```

2. **Select Option 1** - Generate Hybrid Dream11 Teams

3. **Enter Match ID** when prompted:
   ```
   🔍 Enter Match ID: 105780
   ```
   
   **Where to find Match IDs:**
   - Cricbuzz website URLs (e.g., cricbuzz.com/live-cricket-scorecard/105780)
   - Cricket apps and websites
   - Sports news websites
   - Example IDs: 105780, 74648, 86543

4. **Wait for Analysis** (30-60 seconds)
   - The system will auto-fetch all match details
   - Process player statistics
   - Generate optimized teams

5. **Review Results**
   - **Pack-1**: 3 teams with same players, different C/VC
   - **Pack-2**: 2-3 teams with alternative strategies
   - Detailed team breakdowns with credits and scores

### **Sample Output**
```
🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯
🏆 HYBRID TEAM STRATEGY SUMMARY
🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯
📊 Total Teams Generated: 5

📦 PACK-1:
  🏆 Team 1: C: Player A | VC: Player B | Score: 368.3
  🏆 Team 2: C: Player C | VC: Player B | Score: 367.9
  🏆 Team 3: C: Player D | VC: Player B | Score: 365.8

📦 PACK-2:
  🏆 Team 4 (Risk-Adjusted): C: Player A | VC: Player B | Score: 368.3
  🏆 Team 5 (Form-Based): C: Player E | VC: Player F | Score: 301.3
```

## 🛠️ **Troubleshooting**

### **Common Issues & Solutions**

#### **Issue: "No module named 'ortools'"**
**Solution:**
```bash
pip3 install ortools
```

#### **Issue: "API timeout" or "502 Bad Gateway"**
**Solution:**
- The app has built-in fallback data
- Continue with the process - it will use sample data
- Check internet connection

#### **Issue: "No teams generated"**
**Solution:**
- Try a different match ID
- Ensure you're in the correct directory
- Check that all files are present

#### **Issue: "Permission denied" on macOS/Linux**
**Solution:**
```bash
chmod +x run_dreamteam.py
python3 run_dreamteam.py
```

#### **Issue: Python version compatibility**
**Solution:**
```bash
# Check Python version
python3 --version

# If Python < 3.7, update Python
# macOS: brew install python3
# Ubuntu: sudo apt update && sudo apt install python3
```

### **Verification Commands**
```bash
# Test all components
python3 -c "
print('🧪 System Check:')
import sys
print(f'Python: {sys.version.split()[0]}')

try:
    from ortools.linear_solver import pywraplp
    print('✅ OR-Tools: Available')
except:
    print('❌ OR-Tools: Missing')

try:
    import requests
    print('✅ Requests: Available')
except:
    print('❌ Requests: Missing')

try:
    from core_logic import match_resolver
    print('✅ Core Logic: Available')
except:
    print('❌ Core Logic: Missing')
"
```

## 📊 **Performance Information**

- **Analysis Time**: 30-60 seconds per match
- **Teams Generated**: 5-6 teams per run
- **Memory Usage**: ~100MB during processing
- **API Calls**: ~10-15 requests per analysis
- **Optimization**: Uses Google OR-Tools for mathematical optimization

## 🆘 **Support**

If you encounter issues:

1. **Check Prerequisites**: Ensure Python 3.7+ and all dependencies
2. **Verify Files**: All core_logic/ and utils/ files present
3. **Test Components**: Run verification commands above
4. **Check Internet**: API requires internet connectivity
5. **Try Fallback**: App works with sample data if APIs fail

## 📈 **Advanced Usage**

### **Custom Match IDs**
- Find match IDs from cricket websites
- Use recent/upcoming match IDs for best results
- System handles invalid IDs gracefully

### **Batch Processing**
```bash
# Run multiple tests
for match_id in 105780 74648 86543; do
    echo "Testing Match ID: $match_id"
    echo "$match_id" | python3 run_dreamteam.py
done
```

### **Development Mode**
```bash
# Run with debug output
python3 -u run_dreamteam.py | tee dreamteam_output.log
```

---

## ✅ **Quick Start Checklist**

- [ ] Python 3.7+ installed
- [ ] Downloaded/cloned project files
- [ ] Installed dependencies (`pip3 install -r requirements.txt`)
- [ ] Verified installation (test commands passed)
- [ ] Have match ID ready
- [ ] Internet connection available
- [ ] Run `python3 run_dreamteam.py`
- [ ] Select Option 1
- [ ] Enter match ID
- [ ] Wait for results
- [ ] Enjoy optimized teams!

**🎉 You're ready to generate winning Dream11 teams!**