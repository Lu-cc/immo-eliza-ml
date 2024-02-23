# Model card

## Project context

The project involves predicting house prices based on various features such as property characteristics, location, and amenities. The goal is to develop a model that accurately predicts house prices to assist in real estate decision-making processes.

## Data

**Input dataset**: The input dataset consists of property data stored in a CSV file (properties.csv). It includes features such as total area, latitude, longitude, property type, region, and others.

**Target variable**: The target variable is the price of the properties.

**Features**: Features used in the model include numerical features such as total area, latitude, longitude, surface land area, and categorical features such as property type, region, and heating type.

## Model details

Several models were considered during the development process, including linear regression, Lasso regression, Ridge regression and others. Ultimately, a Lasso regression model was chosen as the final model due to its ability to perform feature selection and handle multicollinearity effectively.

## Performance

Performance metrics were evaluated on both training and testing datasets using the R² score. The performance of the Lasso regression model on the training and testing datasets is as follows:

- Train R² score: 0.7274507563289898
- Test R² score: 0.7045226788724903


## Limitations

The model's performance heavily relies on the quality and representativeness of the input data. Biases or errors in the dataset could affect the model's predictions.
The model assumes a linear relationship between features and target variable, which may not always hold true in real-world scenarios.
The model's performance might degrade if applied to datasets with significantly different distributions or features not seen during training.

## Usage

### Dependencies
- click
- pandas
- Pyarrow
- numpy
- scikit-learn
- joblib
- matplotlib

### Training
To train the model, run the train.py script, which preprocesses the data, trains the Lasso regression model, and saves the model artifacts.

### Prediction
To generate predictions, run the predict.py script, providing the path to the input dataset (properties.csv) and the desired output path for predictions.

Example command: 
```python
python predict.py -i "data/properties.csv" -o "output/predictions.csv"
```

## Maintainers

For questions or issues, contact Luca Várhegyi at luca.var@gmail.com.
