# FPL2025 - Fantasy Premier League Team Optimization Tool

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

FPL2025 is an advanced Fantasy Premier League (FPL) team optimization tool that combines data analytics, machine learning, and automated decision-making to help eliminate human bias in team selection and transfers. The project leverages historical data, current season statistics, and predictive modeling to provide data-driven recommendations for maximizing FPL points.

## 🚀 Key Features

### Current Implementation
- **Automated FPL Authentication**: Secure login and session management with the Fantasy Premier League API
- **Real-time Player Data Analysis**: Fetches and processes current season player statistics including goals, assists, points, and advanced metrics like ROI (Return on Investment)
- **Team Analysis & Transfer Suggestions**: Identifies weakest players in your current team and suggests optimal replacements
- **Machine Learning Predictions**: Uses trained models (Extra Trees, LightGBM) to predict player performance and ROI
- **Fixture Difficulty Rating (FDR)**: Custom FDR system based on team form and historical performance
- **Multi-metric Optimization**: Evaluates players using multiple metrics (ROI, points per minute, goal contributions, etc.)
- **Team Constraint Validation**: Ensures all transfers and team selections comply with FPL rules (budget, position limits, team limits)

### Advanced Analytics
- **Historical Data Integration**: Processes and analyzes previous season data for better predictions
- **Form Analysis**: Tracks team form using Win/Draw/Loss patterns and goal differentials
- **Player Performance Metrics**:
  - ROI (Return on Investment): points per million spent
  - Points per game and points per minute efficiency
  - Goal contributions per minute
  - Position-specific performance analysis
  - In depth Fixture analysis with FDR Matrix

## 📁 Project Structure

```
FPL2025/
├── api/                    # Core FPL API integration
│   ├── FPL.py             # Main FPL class with authentication and data fetching
│   └── FPL_helpers.py     # Helper functions for data processing and analysis
├── auth/                  # Authentication module
│   └── fpl_auth.py        # FPL login and session management
├── dataModel/             # Data models and schemas
│   ├── player.py          # Player class with performance calculations
│   ├── team.py            # Team data model
│   ├── fixture.py         # Fixture and match data model
│   └── user.py            # User data model
├── historical/            # Historical data processing
│   ├── historical.py      # Historical data analysis functions
│   └── prepareHistoricalData.py  # Data preparation for training
├── ml/                    # Machine learning components
│   ├── ml.py              # ML model training and prediction
│   └── trainingData.py    # Training data preparation
├── dek/                   # Main execution module
│   └── dek.py             # Current workflow implementation
├── tests/                 # Test files
│   └── test.py            # Basic test functionality
├── constants.py           # API endpoints and configuration constants
├── utils.py               # Utility functions for data conversion
├── requirements.txt       # Python dependencies
└── notes.txt             # Development notes and workflow documentation
```

### Prerequisites
- Python 3.11
- Valid Fantasy Premier League account


## 🔐 Authentication

The tool requires FPL login credentials to access your team data and make transfers. Authentication is handled securely through the `auth/fpl_auth.py` module using the official FPL API endpoints.

**Note**: Keep your credentials secure and never commit them to version control.

## 📊 Current Workflow

### 1. Data Collection & Processing
- Fetches current gameweek and upcoming fixtures
- Retrieves all player statistics and team data
- Calculates custom metrics (ROI, FDR, form analysis)
- Generates CSV datasets for analysis

### 2. Machine Learning Predictions
- Loads pre-trained models (Extra Trees for ROI prediction)
- Predicts player performance for upcoming gameweeks
- Filters top performers based on multiple criteria

### 3. Team Analysis & Optimization
- Analyzes current team performance by position
- Identifies weakest players using multi-metric evaluation
- Suggests optimal transfer candidates from predicted top performers
- Validates all suggestions against FPL constraints

### 4. Transfer Recommendations
- Provides ranked transfer suggestions with improvement metrics
- Calculates cost implications and budget impact
- Ensures team balance and constraint compliance


## 🔮 Upcoming Work & Roadmap

### Short-term Goals (Next Release)
- **Automated Transfer Execution**: Complete implementation of automatic transfer processing
- **Starting XI Optimization**: Algorithm to select best 11 players for each gameweek
- **Captain Selection**: Data-driven captain and vice-captain recommendations
- **UI/Dashboard**: Web interface for easier interaction and visualization
- **Injury & Suspension Tracking**: Real-time player availability monitoring

### Medium-term Features
- **Enhanced Fixture Analysis**: More sophisticated fixture difficulty calculations for future fixtures

### Technical Improvements
- **Model Retraining**: Automated model updates every 5 gameweeks
- **Enhanced Testing**: Comprehensive test suite for all modules
- **Documentation**: Complete API documentation and user guides
- **Performance Optimization**: Faster data processing and reduced API calls
- **Deployment**: Cloud hosting and scheduling capabilities

## 🎯 FPL Constraints & Rules

The tool respects all official FPL constraints:
- **Budget**: £100.0 million total budget
- **Squad Composition**: 2 GK, 5 DEF, 5 MID, 3 FWD
- **Team Limits**: Maximum 3 players from any single Premier League team
- **Starting XI**: 1 GK, minimum 3 DEF, minimum 1 FWD
- **Transfer Limits**: Point deductions for additional transfers

## 📈 Performance Metrics

### Key Performance Indicators
- **ROI (Return on Investment)**: Points earned per million spent
- **Points Per Game**: Average points per appearance
- **Goal Contributions**: Goals + Assists (position adjusted)
- **Form Analysis**: Recent performance trends
- **Fixture Difficulty**: Custom FDR scoring system

### Model Performance
- Current models trained on historical Premier League data
- Multi-metric evaluation for robust player assessment
- Retrains model every 4 gameweeks to avoid potential drift and account for current season bias

## Fixture Difficulty Rating (FDR)

The FDR system is a custom metric that evaluates the difficulty of upcoming fixtures for each team based on their form and historical performance. Here's how it works:

#### Calculation Process:
1. **Team Form Analysis**: Each team's recent form is tracked using Win/Draw/Loss patterns (e.g., "WWDLW")
2. **Form Conversion**: Form strings are converted to numerical values:
   - Win (W) = 3 points
   - Draw (D) = 1 point
   - Loss (L) = 0 points
   - Final form score = total points / (games played × 3) to normalize between 0-1
3. **Fixture Difficulty Calculation**: For each upcoming fixture, the FDR compares:
   - Team's own form score vs opponent's form score
   - Home advantage is considered (team playing at home vs away team)
4. **FDR Score**: Higher positive FDR = easier fixtures, negative FDR = harder fixtures

#### Usage in Predictions:
- Players from teams with easier upcoming fixtures (higher FDR) are more likely to score points
- The FDR is factored into player selection and transfer recommendations
- Helps identify "fixture swings" where previously difficult fixtures become easier

### Score Strength Metric

The Score Strength metric measures a team's ability to win games convincingly by tracking large margin victories and defeats.

#### Calculation Process:
1. **Goal Difference Analysis**: For each completed fixture, if the goal difference is ≥3:
   - Winning team gets +1 score strength
   - Losing team gets -1 score strength
2. **Cumulative Tracking**: Score strength accumulates over the analyzed period
3. **Interpretation**:
   - Positive score strength = team frequently wins by large margins
   - Negative score strength = team frequently loses by large margins
   - Zero/low score strength = team typically involved in close games

#### Current Status:
**Note**: The score_strength metric is currently calculated and tracked but is **not used in the machine learning predictions**. It's available for future implementation and manual analysis but does not influence the current ROI predictions or transfer recommendations.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

### Development Guidelines
- Follow Black code formatting (`black .`)
- Add tests for new features
- Update documentation for API changes
- Respect rate limits when accessing FPL API

## 📝 License

This project is for educational and personal use only. Please respect the Fantasy Premier League terms of service when using this tool.

## 🙏 Acknowledgments

- Fantasy Premier League for providing the official API
- The FPL community for insights and inspiration
- Historical data sources and contributors
---

**Disclaimer**: This tool is intended to assist with FPL decision-making but does not guarantee improved performance. Always use your own judgment and enjoy the game responsibly!