# 🏏 DreamTeamAI - AI-Powered Dream11 Team Predictor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)]()

**DreamTeamAI** is an advanced AI-powered cricket team optimization system that generates winning Dream11 fantasy cricket teams using real-time data analysis, mathematical optimization, and machine learning techniques.

## 🚀 Features

### ⚡ **Hybrid Team Generation Strategy**
- **Pack-1**: Same optimal 11 players with 3 different Captain/Vice-Captain combinations
- **Pack-2**: Alternative teams with different strategies (Risk-Adjusted, Form-Based, Value-Picks)

### 🎯 **Universal Match Support**
- ✅ **Completed Matches**: Historical analysis with actual Playing XI
- ✅ **In-Progress Matches**: Real-time data with live Playing XI
- ✅ **Upcoming Matches**: Squad-based predictions with series data
- ✅ **Special Tournaments**: Champions League, World Cup, etc.

### 🧠 **Advanced Analytics**
- **Exponential Moving Average (EMA)** scoring for recent form
- **Consistency Score** analysis based on performance variance
- **Dynamic Opportunity Index** based on pitch conditions and role
- **Form Momentum** tracking using linear regression
- **Matchup Analysis** against opposition strengths

### 🏟️ **Real-Time Data Integration**
- **Cricbuzz RapidAPI** integration for live match data
- **Automatic Playing XI** extraction for ongoing matches
- **Venue Analysis** with pitch archetype classification
- **Weather and Conditions** impact on player selection

### 🔧 **Mathematical Optimization**
- **Google OR-Tools** integration (SCIP, CBC, CLP, GLOP solvers)
- **Multi-objective optimization** balancing risk and reward
- **Role-based constraints** ensuring balanced team composition
- **Credit optimization** within Dream11 limits

## 📋 Requirements

```txt
requests>=2.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
ortools>=9.4.0
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Dream11_AI.git
   cd Dream11_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run_dreamteam.py
   ```

## 🎮 Usage

### **Method 1: Interactive Menu**
```bash
python run_dreamteam.py
```
- Choose option 1: "Generate Hybrid Dream11 Teams (Match ID)"
- Enter any match ID (completed, in-progress, or upcoming)
- Get hybrid team recommendations in 30-60 seconds

### **Method 2: Direct Match ID**
```python
from run_dreamteam import resolve_match_from_id
from core_logic.data_aggregator import aggregate_all_data
from core_logic.team_generator import generate_hybrid_teams

# Resolve match
match_info = resolve_match_from_id(125217)  # Any match ID
# ... (see examples in code)
```

## 🏏 Match ID Examples

| Match Type | Example ID | Teams | Format |
|------------|------------|-------|--------|
| **Completed** | 114627 | West Indies vs Australia | T20 |
| **In-Progress** | 125217 | India Champions vs WI Champions | T20 |
| **Upcoming** | 105780 | England vs India | TEST |

## 🏗️ Architecture

### **7-Phase Prediction Pipeline**
1. **🔍 Match Resolution** - Fetch match details using Match ID
2. **📊 Data Aggregation** - Gather player statistics and match context
3. **🧠 Feature Engineering** - Calculate performance metrics and predictions
4. **🎯 Base Team Generation** - Create optimal 11-player team
5. **📦 Pack-1 Generation** - Generate C/VC variations of base team
6. **📦 Pack-2 Generation** - Create alternative team strategies
7. **🏆 Results Presentation** - Format and display final recommendations

### **Core Components**
```
Dream11_AI/
├── core_logic/
│   ├── match_resolver.py      # Match ID resolution and data extraction
│   ├── data_aggregator.py     # Player and match data aggregation
│   ├── feature_engine.py      # Advanced analytics and feature generation
│   └── team_generator.py      # Mathematical optimization and team generation
├── utils/
│   └── api_client.py          # Cricbuzz API integration
└── run_dreamteam.py           # Main application interface
```

## 🎯 Key Algorithms

### **Performance Rating Calculation**
```python
performance_score = (
    base_score + 
    historical_performance + 
    opportunity_factor + 
    matchup_factor + 
    form_factor + 
    role_factor
)
```

### **Captain/Vice-Captain Selection**
- **Performance-based ranking** with role diversity
- **Automatic variation** across Pack-1 teams
- **Risk-adjusted selection** for different strategies

### **Team Optimization**
- **Multi-constraint optimization** using OR-Tools
- **Role balance enforcement** (batsmen, bowlers, all-rounders, WK)
- **Credit limit compliance** within Dream11 rules

## 🔧 Configuration

### **API Setup**
The system uses Cricbuzz RapidAPI. Update the API key in `utils/api_client.py`:
```python
API_HEADERS = {
    'x-rapidapi-key': 'YOUR_API_KEY_HERE'
}
```

### **Optimization Settings**
Adjust team generation parameters in `core_logic/team_generator.py`:
- Number of teams per pack
- Risk tolerance levels  
- Captain selection criteria

## 📊 Output Format

### **Pack-1 Teams (Same Players, Different C/VC)**
```
🏆 PACK-1 TEAM 1 - C/VC VARIATION 1
👑 Captain: Yuvraj Singh
🥈 Vice Captain: Piyush Chawla

📋 TEAM COMPOSITION:
🏏 Batsmen (4): Yuvraj Singh, Suresh Raina, Robin Uthappa, Gurkeerat Singh
⚡ Bowlers (4): Piyush Chawla, Harbhajan Singh, Abhimanyu Mithun, Varun Aaron
🔄 All-rounders (2): Stuart Binny, Pawan Negi
🧤 Wicket-keepers (1): Robin Uthappa

📈 DETAILED PLAYER LIST:
  1. Yuvraj Singh        (Batting Allrounder) (C)
  2. Suresh Raina        (Batsman           ) 
  3. Robin Uthappa       (WK-Batsman        ) (VC)
  ...
```

## 🧪 Testing

Run comprehensive tests:
```bash
python comprehensive_test.py
python test_complete_workflow.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and entertainment purposes only. Fantasy sports involves financial risk. Please play responsibly and within your means.

## 🙏 Acknowledgments

- **Cricbuzz API** for real-time cricket data
- **Google OR-Tools** for optimization algorithms
- **scikit-learn** for machine learning utilities
- **pandas & numpy** for data processing

## 📞 Support

For support, issues, or feature requests:
- Open an issue on GitHub
- Contact: [Your Email]

---

**🎉 Generate winning Dream11 teams with AI-powered analysis!**