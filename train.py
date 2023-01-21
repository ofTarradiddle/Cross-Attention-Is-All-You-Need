import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import * 


# Define the model
model = VisionTransformer(dim=512, num_heads=8, num_layers=6)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Define the dataset and dataloader
class GalaxyClustersDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        # Initialize dataset with data from data_path
        self.data = None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item
      
# Define the data augmentation transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply data transformation if needed
transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset
dataset = GalaxyClustersDataset(data_path, transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(dataloader):
        # Forward pass
        output = model(x)
        loss = loss_fn(output, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))

# Validation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in dataloader:
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Accuracy of the model on the validation images: {} %'.format(100 * correct / total))
