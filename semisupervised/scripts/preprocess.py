from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessData(df, numeric_features, categorical_features, target, label_dic):
    """
    Preprocesses a dataset to be directly fed into runExperiment.fit().
    
    Parameters:
    ----------
    df: pandas dataframe
        The dataset.
        
    numeric_features: list(string) or 'remaining'
        List of names of the numeric features. These features are
        scaled. If remaining numeric_features is set to all features
        not in categorical features (in this case categorical features
        cannot be 'remaining').
        
    categorical_features: list(string)  or 'remaining'
        List of names of the categorical features. These features are
        one hot encoded. If remaining numeric_features is set to all features
        not in numeric features (in this case numeric features
        cannot be 'remaining').
        
    target: string
        Name of the target variable.
        
    label_dic: dictionary() or None
        Specifies which values in the target variable should be replaced and
        what they should be replaced by.
        
    Returns:
    --------    
    X: numpy ndarray
        The preprocessed training features.
    y: pandas series
        The target variable.
    """
    
    if numeric_features == 'remaining':    
        numeric_features = [feature for feature in list(df)if
                            feature not in (categorical_features + ['target'])]
        
    if categorical_features == 'remaining':    
        categorical_features = [feature for feature in list(df)if
                            feature not in (numeric_features + ['target'])]
    
    numeric_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps = [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    #not using pipeline because it interferes with semisupervised (bug)
    X = preprocessor.fit_transform(df)
    y = df[target]

    y = y.replace(to_replace = label_dic)
    
    print('nrows: ', len(y))
    print('n_features (post encoding): ', X.shape[1])
    print('target distribution: ', y.value_counts())
    
    return X,y