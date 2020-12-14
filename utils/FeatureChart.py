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
        
        data = data.drop(['id', 'genera'], axis=1)
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
        
        ax = self.data_mean.plot(legend=False, title='Feature Values')
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')