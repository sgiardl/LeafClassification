import numpy as np
import matplotlib.pyplot as plt

class AccuracyChart:
    def __init__(self, names, training_acc, testing_acc, title):
        self.names = names
        self.training_acc = training_acc
        self.testing_acc = testing_acc
        self.width = 0.35
        self.title = title
        
    def display_chart(self):
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
            
        ax.set_ylabel('Accuracy %')
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