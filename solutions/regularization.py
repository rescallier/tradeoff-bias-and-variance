torch.manual_seed(0)
lambda_ = 0.005
cnn_regularization = CNN()
# CrossEntropyLoss as loss because of no softmax in the last layer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_regularization.parameters(),
                             lr=learning_rate,weight_decay=lambda_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## To train the model uncomment lines below, 
#train(cnn_regularization,train_loader,num_epochs,optimizer,criterion)
#torch.save(cnn_regularization, 'models/regularisation.pt')

# Load trained model that was train using the code above using a gpu on google colab during 30 epochs
cnn_regularization = torch.load('models/regularisation.pt')
cnn_regularization.eval()
accuracy_train_regularization = accuracy(cnn_regularization,train_loader,'train')
accuracy_test_regularization  = accuracy(cnn_regularization,test_loader,'test')