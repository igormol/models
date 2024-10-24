# recsysHybrid.py

# Makes 2024 Copyright
# Igor Mol <igor.mol@makes.ai>

# This document contains the hybrid recommender system engine, consisting of a
# Gradient Boosting phase coupled to an Autoencoder neural network.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import xgboost as xgb

def load_data(filepath):
    # Loads data from a CSV file located at the specified filepath
    return pd.read_csv(filepath, parse_dates=['Sales_Date'], dayfirst=True)

def encode_categorical_variables(df, columns):
    # Initializes an empty dictionary to store label encoders
    label_encoders = {}

    # Iterates through each specified column
    for column in columns:
        # Initializes a label encoder for the current column
        label_encoders[column] = LabelEncoder()

        # Transforms the categorical values in the column to numerical values
        df[column] = label_encoders[column].fit_transform(df[column].astype(str))

    # Returns the dataframe with encoded columns and the dictionary of label encoders
    return df, label_encoders

def normalize_numerical_variables(df, columns):
    # Initializes a MinMaxScaler for normalizing numerical variables
    scaler = MinMaxScaler()

    # Applies the scaler to the specified columns in the dataframe
    df[columns] = scaler.fit_transform(df[columns])

    # Returns the dataframe with normalized columns
    return df

def aggregate_duplicates(df, groupby_columns, agg_dict):
    # Groups the dataframe by the specified columns and aggregates using the provided dictionary
    return df.groupby(groupby_columns).agg(agg_dict).reset_index()

def create_interaction_matrix(df, index_col, columns_col, values_col):
    # Creates a pivot table (interaction matrix) with specified index, columns, and values, filling NaNs with 0
    interaction_matrix = df.pivot(index=index_col, columns=columns_col, values=values_col).fillna(0)

    # Converts the interaction matrix to a compressed sparse row matrix
    user_item_matrix = csr_matrix(interaction_matrix.values)

    # Returns the interaction matrix and the sparse matrix
    return interaction_matrix, user_item_matrix

def define_autoencoder(input_dim, encoding_dim):
    # Define a sequential model using a dictionary
    autoencoder = Sequential()

    # Adding the input layer (shape should be specified for the first layer)
    # The Input layer is implicitly created when the first Dense layer is added in Sequential
    autoencoder.add(Dense(encoding_dim, activation="relu", input_shape=(input_dim,)))

    # Adding the decoder layer with the same dimension as the input and sigmoid activation
    # This reconstructs the input from the encoded representation
    autoencoder.add(Dense(input_dim, activation="sigmoid"))

    # Compile the model with Adam optimizer and binary crossentropy loss
    autoencoder.compile(optimizer=Adam(), loss=BinaryCrossentropy())

    # Return the constructed autoencoder model
    return autoencoder

def train_autoencoder(autoencoder, interaction_matrix, epochs, batch_size):
    # Fit the autoencoder model on the interaction matrix
    autoencoder.fit(
        interaction_matrix.values,  # Input data: interaction matrix values
        interaction_matrix.values,  # Output data: same as input for autoencoder
        epochs=epochs,              # Number of training epochs
        batch_size=batch_size,      # Size of each batch for gradient descent
        shuffle=True,               # Shuffle the data before each epoch
        validation_split=0.2        # Use 20% of the data for validation
    )

def get_encoded_representation(autoencoder, interaction_matrix):
    # Create a model that maps inputs to the output of the encoder layer
    encoder_model = Model(
        inputs=autoencoder.input,                   # Input to the autoencoder
        outputs=autoencoder.layers[0].output        # Output of the first layer (encoder output)
    )
    # Use the encoder model to get the encoded representation of the interaction matrix
    return encoder_model.predict(interaction_matrix.values)

def prepare_xgboost_data(train_df, test_df, target_column):
    # Splitting train data into features and target
    X_train = train_df.drop(columns=[target_column])  # Dropping the target column from training data to get features
    y_train = train_df[target_column]  # Extracting the target column as the target variable for training data

    # Splitting test data into features and target
    X_test = test_df.drop(columns=[target_column])  # Dropping the target column from testing data to get features
    y_test = test_df[target_column]  # Extracting the target column as the target variable for testing data

    # Creating DMatrix for XGBoost from training and testing datasets
    train_matrix = xgb.DMatrix(X_train, label=y_train)  # Converting training data to DMatrix
    test_matrix = xgb.DMatrix(X_test, label=y_test)  # Converting testing data to DMatrix

    return train_matrix, test_matrix  # Returning the DMatrix for training and testing data

def train_xgboost_model(dtrain, dtest, params, num_boost_round, early_stopping_rounds):
    # Trains an XGBoost model.
    # Parameters:
    # - dtrain: DMatrix for training data.
    # - dtest: DMatrix for validation data.
    # - params: Dictionary of parameters for XGBoost.
    # - num_boost_round: Number of boosting rounds.
    # - early_stopping_rounds: Rounds of early stopping.
    # Returns:
    # - Trained XGBoost model.

    # Train the XGBoost model using provided parameters, training data, and validation data
    return xgb.train(
        params=params,                # Parameters for the XGBoost model
        dtrain=dtrain,                # DMatrix containing the training data
        num_boost_round=num_boost_round,  # Number of boosting rounds
        evals=[(dtest, 'test')],      # List of evaluation datasets and their names
        early_stopping_rounds=early_stopping_rounds,  # Early stopping rounds
        verbose_eval=False            # Disable verbose evaluation
    )

def get_xgboost_predictions(bst, df_aggregated, target_column):
    # Generates predictions using the trained XGBoost model.
    # Parameters:
    # - bst: Trained XGBoost model.
    # - df_aggregated: DataFrame containing aggregated data.
    # - target_column: Name of the target column in the DataFrame.
    # Returns:
    # - Array of predictions.

    # Create a DMatrix from the aggregated DataFrame, excluding the target column
    df_dmatrix = xgb.DMatrix(df_aggregated.drop(columns=[target_column]))

    # Generate predictions using the trained XGBoost model
    return bst.predict(df_dmatrix)

def reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix):
    # Initialize an empty matrix with the same shape as the interaction matrix
    xgb_predictions_matrix = np.zeros(interaction_matrix.shape)

    # Iterate over the predictions and corresponding client and product IDs
    for i, (client_id, product_id) in enumerate(zip(df_aggregated['Client_ID'], df_aggregated['Product_ID'])):
        # Find the row index in the interaction matrix for the current client ID
        client_idx = interaction_matrix.index.get_loc(client_id)

        # Find the column index in the interaction matrix for the current product ID
        product_idx = interaction_matrix.columns.get_loc(product_id)

        # Assign the XGBoost prediction to the appropriate position in the prediction matrix
        xgb_predictions_matrix[client_idx, product_idx] = xgb_predictions[i]

    # Return the reshaped predictions matrix
    return xgb_predictions_matrix

def combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix):
    # Predict the interactions using the autoencoder model
    autoencoder_predictions = autoencoder.predict(interaction_matrix.values)

    # Combine the autoencoder predictions and XGBoost predictions by averaging them
    combined_predictions = (autoencoder_predictions + xgb_predictions_matrix) / 2

    # Convert the combined predictions to a DataFrame with the same index and columns as the interaction matrix
    return pd.DataFrame(combined_predictions, index=interaction_matrix.index, columns=interaction_matrix.columns)

def get_top_k_recommendations(combined_df, interaction_matrix, user_id, k=5):
    # Find the row index in the interaction matrix corresponding to the specified user ID
    user_row_number = interaction_matrix.index.get_loc(user_id)

    # Sort the predictions for this user in descending order
    sorted_user_predictions = combined_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the top k product IDs and their corresponding predicted ratings
    top_k_products = sorted_user_predictions.head(k).index.tolist()
    top_k_scores = sorted_user_predictions.head(k).tolist()

    # Return the top k product IDs and their predicted ratings
    return list(zip(top_k_products, top_k_scores))

def main():
    # Load the dataset
    df = load_data('Marajo.csv')

    # Print the first few rows of the dataset
    print("Data loaded successfully:")
    print(df.head())

    # Encode categorical variables
    df, label_encoders = encode_categorical_variables(df, ['Client_ID', 'Product_ID', 'Sales_Origin', 'Segment'])

    # Fill missing values in 'Brand_Name' with 'Unknown'
    df['Brand_Name'].fillna('Unknown', inplace=True)

    # Encode the 'Brand_Name' column
    df, brand_name_encoder = encode_categorical_variables(df, ['Brand_Name'])

    # Normalize numerical variables
    df = normalize_numerical_variables(df, ['Qtd', 'Volume_m3', 'Weight_kg', 'Price_BR'])

    # Print the first few rows of the preprocessed dataset
    print("Data preprocessed successfully:")
    print(df.head())

    # Aggregate duplicates
    df_aggregated = aggregate_duplicates(df, ['Client_ID', 'Product_ID'], {'Qtd': 'sum'})

    # Print the first few rows of the aggregated dataset
    print("Data aggregated successfully:")
    print(df_aggregated.head())

    # Create interaction matrix
    interaction_matrix, user_item_matrix = create_interaction_matrix(df_aggregated, 'Client_ID', 'Product_ID', 'Qtd')

    # Define autoencoder parameters
    input_dim = user_item_matrix.shape[1]
    encoding_dim = 64

    # Define and train the autoencoder
    autoencoder = define_autoencoder(input_dim, encoding_dim)
    train_autoencoder(autoencoder, interaction_matrix, epochs=50, batch_size=256)

    # Get the encoded representation of the interaction matrix
    encoded_representation = get_encoded_representation(autoencoder, interaction_matrix)

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df_aggregated, test_size=0.2, random_state=42)

    # Prepare data for XGBoost
    train_matrix, test_matrix = prepare_xgboost_data(train_df, test_df, 'Qtd')

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1
    }

    # Train the XGBoost model
    bst = train_xgboost_model(train_matrix, test_matrix, params, num_boost_round=100, early_stopping_rounds=10)

    # Generate XGBoost predictions
    xgb_predictions = get_xgboost_predictions(bst, df_aggregated, 'Qtd')

    # Reshape XGBoost predictions
    xgb_predictions_matrix = reshape_xgboost_predictions(xgb_predictions, df_aggregated, interaction_matrix)

    # Combine predictions
    combined_df = combine_predictions(autoencoder, interaction_matrix, xgb_predictions_matrix)

    # Get top 5 recommendations for a specific user
    user_id = 1  # Specify the user ID
    top_k_recommendations = get_top_k_recommendations(combined_df, interaction_matrix, user_id, k=5)

    # Print the top k recommendations
    print(f"Top {5} recommendations for user {user_id}:")
    for product_id, score in top_k_recommendations:
        print(f"Product ID: {product_id}, Predicted Score: {score}")

if __name__ == "__main__":
    main()
