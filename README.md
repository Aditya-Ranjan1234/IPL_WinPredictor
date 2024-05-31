# IPL_WinPredictor

## Project Overview

IPL_WinPredictor is a machine learning project designed to predict the probability of a team winning or losing a match in the Indian Premier League (IPL). Using historical IPL data from 2008 to 2019, the model analyzes various factors such as team performance, match conditions, and current match status to provide real-time win/loss probabilities during a game.

## Dataset

The dataset used for this project includes IPL match data from 2008 to 2019. The dataset consists of various features such as:

- `match_id`: Unique identifier for each match
- `batting_team`: The team currently batting
- `bowling_team`: The team currently bowling
- `city`: The city where the match is being played
- `runs_left`: Runs required to win the match
- `balls_left`: Balls remaining in the innings
- `wickets`: Wickets lost by the batting team
- `total_runs_x`: Target runs set by the opponent team
- `crr`: Current run rate
- `rrr`: Required run rate

## Model

The model used in this project is RandomForestClassifier that predicts the win/loss probabilities for the batting team. The model is trained on historical IPL data and uses features such as current score, overs completed, wickets lost, and match location to make predictions.

### Features

- **batting_team**: The team currently batting.
- **bowling_team**: The team currently bowling.
- **city**: The city where the match is being played.
- **runs_left**: The number of runs left to achieve the target.
- **balls_left**: The number of balls left in the innings.
- **wickets**: The number of wickets remaining for the batting team.
- **total_runs_x**: The target score set by the opponent team.
- **crr**: Current run rate of the batting team.
- **rrr**: Required run rate to achieve the target.

## Usage

**Prepare Input Data**: Prepare a DataFrame with the input features as shown in the example below:

    # Sample input data
    batting_team = 'Chennai Super Kings'
    bowling_team = 'Mumbai Indians'
    selected_city = 'Mumbai'
    target = 180
    score = 12
    overs = 2.2

    # Calculations
    runs_left = target - score
    balls_left = 120 - (int(overs) * 6 + (overs - int(overs)) * 10)
    wickets = 10 - 2  # Assuming 2 wickets down
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create DataFrame
    temp_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Load the model (assuming pipe is the trained model)
    result = pipe.predict_proba(temp_df)

    # Add predicted probabilities to DataFrame
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)

    # Display the results
    temp_df = temp_df[['lose', 'win']]
    print(temp_df)
    ```

**Interpret the Results**: The output DataFrame will provide the probabilities of winning and losing for the batting team.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn (or other machine learning libraries used for training the model)

## Installation

Clone the repository:
    ```bash
    git clone https://github.com/Aditya-Ranjan1234/IPL_WinPredictor.git
    ```

## Credits

This code was originally created by CampusX (@campusx-official). Various other models have been added and modified.
