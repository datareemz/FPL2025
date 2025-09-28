# FPL2025 - Fantasy Premier League bot

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

FPL2025 is an autonomous Fantasy Premier League bot that automatically manages your FPL team using data analytics, machine learning, and automated decision-making. Hosted on Google Cloud, the bot eliminates human bias by making transfers, selecting lineups, and choosing captains based on predictive modeling and current season statistics. The system runs scheduled jobs to continuously optimize your team for maximum points without manual intervention.

## 🚀 Key Features
- **Automated FPL Authentication**: Secure login and session management with the Fantasy Premier League API
- **Real-time Player Data Analysis**: Fetches and processes current season player statistics including goals, assists, points, and advanced metrics like ROI (Return on Investment)
- **Team Analysis & Transfer Suggestions**: Identifies weakest players in your current team and suggests optimal replacements
- **Machine Learning Predictions**: Uses trained models (Extra Trees) to predict player performance and ROI
- **Multi-metric Optimization**: Evaluates players using multiple metrics (ROI, points per minute, goal contributions, etc.)
- **Team Constraint Validation**: Ensures all transfers and team selections comply with FPL rules (budget, position limits, team limits)
- **Historical Data Integration**: Processes and analyzes previous season data for better predictions


## ☁️ Google Cloud Infrastructure
The project uses Google Cloud Platform for automated deployment and scheduling:
- **Cloud Run**: Containerized job execution
- **Cloud Scheduler**: Automated job scheduling
- **Artifact Registry**: Docker image storage
- **Cloud Build**: CI/CD pipeline integration
- **IAM Service**: GitHub Actions authentication

Terraform manages all GCP resources with modular configurations for each service.

### Prerequisites
- Python 3.11
- Valid Fantasy Premier League account
- Google Cloud Platform account (for deployment)

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
- Predicts player performance for upcoming gameweeks using generated csv
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
- **Injury & Suspension Tracking**: Real-time player availability monitoring
- **Automated Transfer Execution**: Complete implementation of automatic transfer processing
- **Starting XI Optimization**: Algorithm to select best 11 players for each gameweek
- **Captain Selection**: Data-driven captain and vice-captain recommendations
- **UI/Dashboard**: Web interface for easier interaction and visualization


### Medium-term Features
- **Enhanced Fixture Analysis**: More sophisticated fixture difficulty calculations for future fixtures


## 🎯 FPL Constraints & Rules
https://fantasy.premierleague.com/help/rules


## ML features
- **ROI (Return on Investment)**: Points earned per million spent
- **Points Per Game**: Average points per appearance
- **Goal Contributions**: Goals + Assists
- **ROI per Gameweek**: ROI per matchday
- **Form Analysis**: Recent performance trends
- **Fixture Difficulty Rating**: Custom FDR scoring system

## Model Performance
- Model performance will be measured and shared online (TODO)

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


### Contributing/Development Guidelines (Incoming...)
- Follow Black code formatting (`black .`)
- Add tests for new features
- Update documentation where necessary

## 🙏 Acknowledgments
- Fantasy Premier League for providing the official API
- The FPL community for insights and inspiration
- Historical data sources and contributors
- @vaastav, @amosbastian

**Disclaimer**: This tool is intended to assist with FPL decision-making but does not guarantee improved performance. Always use your own judgment and enjoy the game responsibly!