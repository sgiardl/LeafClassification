class FeatureChart:
    def __init__(self, data):
        data = data.drop(['id', 'genera'], axis=1)
        self.data_mean = data.groupby('species').mean().T
        
    def display_chart(self):
        ax = self.data_mean.plot(legend=False, title='Feature Values')
        ax.set_xlabel('Features')
        ax.set_ylabel('Values')