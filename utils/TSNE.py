from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

class t_SNE:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def display_TSNE(self):
        tsne = TSNE(init='pca')
        output = tsne.fit_transform(self.X, self.y)

        df = pd.DataFrame({"x": output[:, 0],
                           "y": output[:, 1],
                           "colors": self.y})
        
        df = df.sort_values(by=['colors'])
        
        symbols = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
      
        self.__display_TSNE_chart(df, symbols, 'TSNE - No Grouping')

        tsne_groups = [[57,17,44],[62,66,27,97,77],[89,91,18,60],[13,95,11,14],
                       [88,7,26,58],[16,8,1],[93,12,41,48],[49,75],
                       [92,73,86],[81,42,83,52],[50,53,56,3],[94,10,98,47,19],
                       [74,20,0,9,65],[85,54,76,],[68,79,46,45],[43,96,24],
                       [21,34,30],[28,29,37,38],[23,25,15],[39,40,33],
                       [70,72],[2,5,35],[6,78],[82,59,63,80,71],[31],[84,4],
                       [36,22,67],[32,61,87],[64,51,90,55,69]]
        
        i = 0
        
        tsne_groups_dict = {}
        
        for group in tsne_groups:      
            for no in group:
                tsne_groups_dict[no] = i
                
            i += 1   
            
        df = df.replace({'colors':tsne_groups_dict})

        self.__display_TSNE_chart(df, symbols, 'TSNE - With Grouping')
    
        df = df.sort_index()
        
        self.y = df['colors']
        
    def __display_TSNE_chart(self, df, symbols, title):
        plt.figure()
        
        cmap = plt.cm.Spectral
        norm = plt.Normalize(df['colors'].values.min(), df['colors'].values.max())

        for i, dff in df.groupby('colors'):
            plt.scatter(dff['x'], dff['y'], c=cmap(norm(dff['colors'])),
                        s=80, marker=symbols[i % len(symbols)], label=f'{i}')
            
        plt.legend(ncol=4)
        plt.title(title)
        plt.show()    
        

        
        
        
        
        
        
        