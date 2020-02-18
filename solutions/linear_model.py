class CNN(nn.Module):
    '''
    Define the model 
    '''
    def __init__(self):
        super(CNN, self).__init__()
        
        self.fc = nn.Linear(28*28*1, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out