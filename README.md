# Banking-Stock-Prediction-Using-Feature-Selection-and-Hyperparameter-Tuning_Thesis

This research aims to predict stock prices by combining a Long Short-Term Memory (LSTM) model with a Genetic Algorithm (GA) for hyperparameter optimization and feature selection. The dataset includes historical stock data and technical indicators from April 2010 to April 2025.

Research Procedure
1. ## Data Collection ##: Collect historical stock prices and technical indicators covering the period April 2010 – April 2025.
2. ## Data Preparation
a. Perform dimensionality reduction by removing highly correlated variables to avoid redundancy.
b. Handle missing values.
c. Align the size and structure of all variables to ensure consistent processing.
3. ## Data Normalization
Apply Min-Max Scaling to normalize all input variables and standardize their value ranges.
4. Data Splitting: Split data using the sliding window method with the following proportions: 80% training data, 10% validation data, 10% testing data
5. Building the LSTM + GA Model
a. Design the base network architecture consisting of: 2 LSTM layers, 1 dropout layer, 1 dense layer
b. Initialize the Genetic Algorithm population, where each individual represents: Number of neurons in LSTM Layer 1 and Layer 2, Dropout rate, Number of training epochs, Selected feature subset
c. For each individual, train the model on training data and compute its fitness based on the minimum validation loss achieved.
6. Genetic Algorithm Evolution: GA performs selection, crossover, and mutation until reaching the final generation.
7. Selecting the Best Model: The individual with the lowest validation loss is chosen as the optimal model.
8. Model Evaluation: Evaluate the final model using testing data with MAPE (Mean Absolute Percentage Error) and RMSE (Root Mean Squared Error)
9. Stock Price Prediction: The best model obtained through GA optimization is used to predict stock prices for the next one month.
