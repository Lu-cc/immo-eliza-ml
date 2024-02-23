import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = [
        "total_area_sqm",
        "latitude",
        "longitude",
        "surface_land_sqm",
        "garden_sqm",
        "primary_energy_consumption_sqm",
        "construction_year",
        "cadastral_income",
        "nbr_frontages",
        "nbr_bedrooms",
        "terrace_sqm",
    ]
    fl_features = [
        "fl_garden",
        "fl_furnished",
        "fl_open_fire",
        "fl_terrace",
        "fl_swimming_pool",
        "fl_floodzone",
        "fl_double_glazing",
    ]
    cat_features = [
        "property_type",
        "subproperty_type",
        "region",
        "province",
        "locality",
        "state_building",
        "epc",
        "heating_type",
        "equipped_kitchen",
    ]

    #Outliers handling
    Q1 = data["price"].quantile(0.25)
    Q3 = data["price"].quantile(0.75)
    IQR = Q3 - Q1

    max_value = Q3 + (1.5 * IQR)
    min_value = Q1 - (1.5 * IQR)
    
    outliers_mask = (data["price"] < min_value) | (data["price"] > max_value)
    data.loc[outliers_mask, "price"] = np.nan

    data.dropna(subset=["price"], inplace=True)
    
    #Drop missing values from cat_features
    data[cat_features] = data[cat_features].replace("MISSING", None)
    data[cat_features].dropna() 

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Standardizing the numerical features
    scaler = StandardScaler()
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    model_lasso = Lasso(alpha=0.1)  # regularization strength
    model_lasso.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model_lasso.predict(X_train))
    test_score = r2_score(y_test, model_lasso.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model_lasso,
        "scaler": scaler,
    }
    joblib.dump(artifacts, "models/artifacts_lasso.joblib")


if __name__ == "__main__":
    train()
