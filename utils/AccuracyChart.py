import numpy as np
import matplotlib.pyplot as plt

class AccuracyChart:
    """
    CLASS NAME:
        AccuracyChart
        
    DESCRIPTION:
        This class creates a bar chart comparing
        the training and testing accuracies of all
        tested classifying methods.
    """
    def __init__(self, names, training_acc, testing_acc, title):
        """
        PARAMETERS:
            names : list of name of each classifier, list of string
            training_acc : list of training accuracies, list of float
            testing_acc : list of testing accuracies, list of float
            title : title name for the chart, string
                
        RETURNS:
            None.
            
        DESCRIPTION:
            Initializes the AccuracyChart class, passes the arguments
            names, training_acc, testing_acc and title as member
            arguments of the class and specifies the member argument
            width of the class as 0.35 for the bar width.
        """
        
        self.names = names
        self.training_acc = training_acc
        self.testing_acc = testing_acc
        self.width = 0.35
        self.title = title
        
    def display_chart(self):
        """
        PARAMETERS:
            None.
            
        RETURNS:
            None.
            
        DESCRIPTION:
            This method displays the accuracy chart using matplotlib.
        """
        
        # Format the bar chart axes and figure
        x = np.arange(len(self.names))
        fig, ax = plt.subplots()
        ax.bar(x - self.width/2, 
               self.training_acc, 
               self.width, 
               label='Training')
        ax.bar(x + self.width/2, 
               self.testing_acc, 
               self.width, 
               label='Testing')
            
        # Set y label for the bar chart
        ax.set_ylabel('Mean Accuracy %')
        
        # Set x labels for the bar chart
        ax.set_xticks(x)
        ax.set_xticklabels(self.names, rotation='vertical')
        
        # Show the legend
        ax.legend()
        
        # Adjust the y range on the bar chart
        plt.ylim([min(self.training_acc + self.testing_acc) - 1, 
                  max(self.training_acc + self.testing_acc)])
        
        # Add a grid on the bar chart
        plt.grid(axis='y', linestyle='--')
        
        # Fit the plot into a tight layout4
        plt.tight_layout()
        
        # Set title for the bar chart
        plt.title(self.title)
        
        # Show the plot
        plt.show()        