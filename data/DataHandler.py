import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    """
    CLASS NAME:
        DataHandler
    
    DESCRIPTION:
        This class is used to read the input database and manage
        the data, which then can be sent to the classifiers.
    """
    def __init__(self, path):  
        """
        PARAMETERS:
            path : file path to the train.csv file        
        
        RETURNS:
            None.
            
        DESCRIPTION:
            The data in the csv file located in the specified path
            is read and saved in the attributes self.data and self.X.
            
            self.data : the original data with all features and labels,
                        pandas dataframe
            self.X : the X data with no labels, numpy array
        """
        
        # Read the csv file specified in path
        self.data = pd.read_csv(path)
        
        # Add a new column to the data pandas dataframe using the
        # genera text found before the first _ in the species
        # column
        self.data['genera'] = self.data.species.str.split('_').str[0]
        
        # Store the X values as a numpy array, removing the
        # 'species', 'id' and 'genera' columns
        self.X = np.array(self.data.drop(['species', 'id', 'genera'], axis=1) )
    
    def get_y(self, label_col):
        """
        PARAMETERS:
            label_col : column containing the labels in self.data,
                        pandas dataframe column, either
                        self.data.species to group classes by species
                        or self.data.genera to group classes by genera
        
        RETURNS:
            y : labels encoded into numerical values, numpy array
            
        DESCRIPTION:
            The labels in the label_col column are encoded into
            numerical values and the function returns these
            numerical values.
        """
        
        # Transform the labels into numerical values
        return LabelEncoder().fit(label_col).transform(label_col) 
    
    def split_data(self, X, y, test_size, norm):
        """
        PARAMETERS:
            X : features data, 2D numpy array
            y : labels data, 1D numpy array
            test_size : size of test set as floating point value (ex. 0.2 for 20%)
            norm : normalizing method to use for X, choices :
                'none' : no normalizing
                'min-max': normalizing using the min-max method
                'mean': normalizing using the mean-std deviation method
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method splits the X and y data into stratified shuffled
            K-fold splits for training and testing sets. All splits are
            saved in the following list member attributes of the class,
            as lists of numpy arrays:
                self.X_train_list
                self.X_test_list
                self.y_train_list
                self.y_test_list
        
        """
        
        # Calculate number of splits (inverse of test_size)
        n_splits = int(1 / test_size)
        self.n_splits = n_splits
        
        # Declaring the StratifiedKFold object
        stratified_split = StratifiedKFold(n_splits=n_splits,
                                           shuffle=True,
                                           random_state=0)
  
        # Declaring empty lists to store each split data and labels array
        self.X_train_list = []
        self.X_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        
        # Main loop to generate stratified K-Fold splits of the
        # training and testing data
        for index_train, index_test in stratified_split.split(X, y):
            # Get X values based on generated indices
            X_train, X_test = X[index_train], X[index_test]
            
            # Get y values based on generated indices
            y_train, y_test = y[index_train], y[index_test]
        
            # Calculate the length of the training and testing datasetss
            train_size = len(X_train)
            test_size = len(X_test)
            
            # Apply the specified normalization method on the X data
            if norm == 'min-max':
                # Calculate the minimum column-wise values for the training data
                min_norm_train = self.__make_array(X_train.min(axis=0), train_size)
                
                # Calculate the maximum column-wise values for the training data                
                max_norm_train = self.__make_array(X_train.max(axis=0), train_size)   
                
                # Calculate the minimum column-wise values for the testing data                
                min_norm_test = self.__make_array(X_train.min(axis=0), test_size)
                
                # Calculate the maximum column-wise values for the testing data                
                max_norm_test = self.__make_array(X_train.max(axis=0), test_size)          
                
                # Normalize the training data using the min-max method
                X_train = self.__normalize_min_max(X_train, min_norm_train, max_norm_train)
               
                 # Normalize the testing data using the min-max method
                X_test = self.__normalize_min_max(X_test, min_norm_test, max_norm_test)
                
            elif norm == 'mean':
                # Calculate the mean column-wise values for the training data
                mean_norm_train = self.__make_array(X_train.mean(axis=0), train_size)
                
                # Calculate the standard deviation column-wise values for the training data
                std_norm_train = self.__make_array(X_train.std(axis=0), train_size) 
                
                # Calculate the mean column-wise values for the testing data
                mean_norm_test = self.__make_array(X_train.mean(axis=0), test_size)
                
                # Calculate the standard deviation column-wise values for the testing data
                std_norm_test = self.__make_array(X_train.std(axis=0), test_size)  
                
                # Normalize the training data using the mean method
                X_train = self.__normalize_mean(X_train, mean_norm_train, std_norm_train)
                
                # Normalize the testing data using the mean method                
                X_test = self.__normalize_mean(X_test, mean_norm_test, std_norm_test)            
            
            # Add the training and testing data sets into the data set lists
            self.X_train_list.append(X_train)
            self.X_test_list.append(X_test)
            self.y_train_list.append(y_train)
            self.y_test_list.append(y_test)
    
    def __make_array(self, X, size):
        """
        PARAMETERS:
            X : 1D numpy array of column-wise normalization parameters
            size : number of times to duplicate X, integer
        RETURNS:
            array : 2D numpy array of wanted size
            
        DESCRIPTION:
            This private method takes a horizontal numpy array X
            and duplicates it 'size' times vertically to convert a 1D
            vector X to a 2D array of height 'size'
        """
        
        # Convert 1D horizontal array to 2D array of height 'size'
        return np.array([X] * size)
    
    def __normalize_min_max(self, X, X_min, X_max):
        """
        PARAMETERS:
            X : values, 2D numpy array
            X_min : column-wise minimum values, 2D numpy array
            X_max : column-wise maximum values, 2D numpy array
                
        RETURNS:
            x_norm : normalized values, 2D numpy array
            
        DESCRIPTION:
            This private method normalizes the X values and
            returns a new array with normalized values using
            the 'min-max' normalization method
        """
        
        # Apply max-min normalization array-wise
        return (X - X_min) / (X_max - X_min)

    def __normalize_mean(self, X, X_mean, X_std):
        """
        PARAMETERS:
            X : values, 2D numpy array
            X_mean : column-wise mean values, 2D numpy array
            X_std : column-wise standard deviation values, 
                    2D numpy array
                
        RETURNS:
            x_norm : normalized values, 2D numpy array
            
        DESCRIPTION:
            This private method normalizes the X values and
            returns a new array with normalized values using
            the 'mean' normalization method
        """
        
        # Apply mean normalization array-wise        
        return (X - X_mean) / X_std