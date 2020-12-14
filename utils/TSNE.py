from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class t_SNE:
    """
    CLASS NAME:
        t_SNE
        
    DESCRIPTION:
        This class is used to generate a t-distributed stochastic 
        neighbor embedding (t-SNE) chart to show the class similarity
        for all features in a 2D chart.
    """
    def __init__(self, X, y):
        """
        PARAMETERS:
            X : features data, 2D numpy array
            y : labels, 1D numpy array
        RETURNS:
            None.
            
        DESCRIPTION:
            Saves the arguments as member attributes
            of the class self.X and self.y.
        """
        
        self.X = X
        self.y = y
        
    def display_TSNE(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method displays the t-SNE chart with
            and without class grouping.
        """
        
        # Initialize the t-SNE object
        tsne = TSNE(init='pca')
        
        # Get the t-SNE points
        output = tsne.fit_transform(self.X, self.y)

        # Store the t-SNE points into a pandas dataframe
        df = pd.DataFrame({"x": output[:, 0],
                           "y": output[:, 1],
                           "colors": self.y})
        
        # Sort the dataframe values by color
        df = df.sort_values(by=['colors'])
        
        # Declare symbols list for the t-SNE chart
        symbols = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
      
        # Display the t-SNE chart without grouping any classes
        self.__display_TSNE_chart(df, symbols, 'TSNE - No Grouping')

        # Declare a list of the t-SNE groups, each sub-list
        # containing the class IDs that are grouped together
        tsne_groups = [[57,17,44],[62,66,27,97,77],[89,91,18,60],[13,95,11,14],
                       [88,7,26,58],[16,8,1],[93,12,41,48],[49,75],
                       [92,73,86],[81,42,83,52],[50,53,56,3],[94,10,98,47,19],
                       [74,20,0,9,65],[85,54,76,],[68,79,46,45],[43,96,24],
                       [21,34,30],[28,29,37,38],[23,25,15],[39,40,33],
                       [70,72],[2,5,35],[6,78],[82,59,63,80,71],[31],[84,4],
                       [36,22,67],[32,61,87],[64,51,90,55,69]]
        
        # Reset new class colors to be sequential starting from 0 
        i = 0
        
        tsne_groups_dict = {}
        
        for group in tsne_groups:      
            for no in group:
                tsne_groups_dict[no] = i
                
            i += 1   
            
        df = df.replace({'colors':tsne_groups_dict})

        # Display the t-SNE chart grouping similar classes
        self.__display_TSNE_chart(df, symbols, 'TSNE - With Grouping')
    
        # Sort the dataframe values by index
        df = df.sort_index()
        
        # Save the colors as the self.y attribute
        self.y = df['colors']
        
    def __display_TSNE_chart(self, df, symbols, title):
        """
        PARAMETERS:
            df : pandas dataframe containing the results of the
                 t-SNE algorithms and color values in 3 columns :
                 'x', 'y' and 'colors'
            symbols : list of symbols to use for the t-SNE chart, 
                      list of strings
            title : title for the t-SNE chart, string
                
        RETURNS:
            None.
            
        DESCRIPTION:
            This private method is a sub-method called twice
            in the display_TSNE() method. It generates a t-SNE
            chart for the data contained in the df argument.
        """
        
        # Initialize plot figure
        plt.figure()
        
        # Declare color and normalization maps
        cmap = plt.cm.Spectral
        norm = plt.Normalize(df['colors'].values.min(), df['colors'].values.max())

        # Add a scatter plot group for each color group
        for i, dff in df.groupby('colors'):
            plt.scatter(dff['x'], dff['y'], c=cmap(norm(dff['colors'])),
                        s=80, marker=symbols[i % len(symbols)], label=f'{i}')
            
        # Add a legend to the t-SNE chart
        plt.legend(ncol=4)
        
        # Add a title to the t-SNE chart
        plt.title(title)
        
        # Show the t-SNE chart
        plt.show()    