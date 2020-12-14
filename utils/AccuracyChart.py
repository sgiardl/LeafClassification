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
            
        ax.set_ylabel('Mean Accuracy %')
        ax.set_title('Training and testing accuracies')
        ax.set_xticks(x)
        ax.set_xticklabels(self.names, rotation='vertical')
        ax.legend()
        plt.ylim([min(self.training_acc + self.testing_acc) - 1, 
                  max(self.training_acc + self.testing_acc)])
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.title(self.title)
        plt.show()        