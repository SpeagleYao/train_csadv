import torch
import torch.nn as nn
import torch.nn.functional as F

class cst_mat_loss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(cst_mat_loss, self).__init__()
        self.cost_matrix = torch.Tensor([
           # 0  1  2  3  4  5  6  7  8  9
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 0
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 1
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 2
            [1, 1, 1, 1, 1, 5, 1, 1, 1, 1], # 3
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 4
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 5
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 6
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 7
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 9
        ]).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.getTarget()

    def getLabel(self, index):
        if index == 0:
            return "airplane"
        if index == 1:
            return "automobile"
        if index == 2:
            return "bird"
        if index == 3:
            return "cat"
        if index == 4:
            return "deer"
        if index == 5:
            return "dog"
        if index == 6:
            return "frog"
        if index == 7:
            return "horse"
        if index == 8:
            return "ship"
        if index == 9:
            return "truck"

    def forward(self, x, target):
        
        cost_matrix_pick = self.cost_matrix[target, torch.argmax(x, dim=1)]
        loss = F.cross_entropy(x, target, reduction='none') * cost_matrix_pick

        return loss.mean()

    def getTarget(self):

        for i in range(10):
            for j in range(10):
                if self.cost_matrix[i][j] != 1:
                    print("Weight value from " + self.getLabel(i) + " to " + self.getLabel(j) + " = " + str(self.cost_matrix[i][j].cpu().numpy()))



    


        '''
        self.cost_matrix = torch.Tensor([
           # 0  1  2  3  4  5  6  7  8  9
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 0
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 1
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 2
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 3
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 4
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 5
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 6
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 7
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 8
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 9
        ]).cuda()
        '''