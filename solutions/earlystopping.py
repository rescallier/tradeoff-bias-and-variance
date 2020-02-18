def train(cnn,train_loader,num_epochs,optimizer,criterion,validloader,patience):
    '''
    Train the model
    -------
    
    Param:
        cnn : torch.nn.module, model to train
        train_loader : torch.utils.data.DataLoader, loader with the data to train the model on
        num_epochs : int, number of epoch 
        optimizer : torch.optim, optimizer to use during the training
        criterion: torch.nn, loss function used here,
        validloader: torch.utils.data.DataLoader, loader with the data to validate the model on
        patience: int, number of epoch to wait with an higher loss before stopping the algorithm
    '''
    losses = []
    validlosses = []
    estop = EarlyStopping(patience=patience)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): 
        
            images = Variable(images.float())
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.data);
            
            if (i+1) % 100 == 0:
                  print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.detach().numpy()))
        valid_loss = validation(cnn,validloader,criterion)
        validlosses.append(valid_loss)
        estop.step(valid_loss)
        print ('Valid Loss, Epoch : %d/%d,  Loss: %.4f' 
                    %(epoch+1, num_epochs, valid_loss))
        if estop.early_stop:
            break         