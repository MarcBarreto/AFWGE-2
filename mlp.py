import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def fit(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        total = 0
        accuracy = .0
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            
            optimizer.zero_grad()
            
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            _, predicted = torch.max(outputs.data, axis = 1)
            total += labels.size(0)

            accuracy += (predicted == labels).sum().item()

            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/total:.4f} Accuracy: {accuracy/total:.2f}')

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total}%')

def inference(model, data):
    model.eval()

    with torch.no_grad():
        if len(data.shape) == 1:
            data.squeeze(0)
        output = model(data)
        _, result = torch.max(output.data, 1)
        
        return result

def train_model(X, y, train_loader, test_loader, test = False):
    input_size = X.shape[1]
    output_size = len(set(y))
    learning_rate = 0.001
    epochs = 30

    model = MLP(input_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    fit(model, train_loader, criterion, optimizer, epochs)
    
    if test:
        test_model(model, test_loader)
    
    return model