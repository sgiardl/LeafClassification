class FeatureChart:
    """
    CLASS NAME:
        FeatureChart
        
    DESCRIPTION:
        This class creates a chart showing the range
        of possible values for each data feature.
    """
    def __init__(self, data):
        """
        PARAMETERS:
            data : data pandas dataframe, as found in the 
                   DataHandler.data attribute   
                
        RETURNS:
            None.
            
        DESCRIPTION:
            Loads the raw data from the DataHandler class object,
            processes it, calculates the mean value for each
            species class and transposes it so the X axis is the
            features and the y axis is the mean value for each 
            species.
        
        """
        # Remove the 'id' and 'genera' columns from the data
        # pandas dataframe        
        data = data.drop(['id', 'genera'], axis=1)
        
        # Calculate the mean feature values for each species
        # and transpose the data so the features are on the X
        # axis and the y axis shows the mean value for each
        # species
        self.data_mean = data.groupby('species').mean().T
        
    def display_chart(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            Displays a chart showing the range of values
            for each feature.   
        """
        
        # Display the features chart, specify that
        # to remove the legend and add a title
        ax = self.data_mean.plot(legend=False, title='Feature Values')
        
        # Set the X axis label to 'Features'
        ax.set_xlabel('Features')
        
        # Set the y axis label to 'Values'
        ax.set_ylabel('Values')