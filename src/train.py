import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm  

# 1. Thiết lập thiết bị (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Chuẩn bị dữ liệu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = False)

# 3. Định nghĩa mạng nơ-ron
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # Chuyển ảnh 28x28 thành vector 784
        self.layer1 = nn.Linear(28*28, 512)  # Lớp fully connected đầu tiên
        self.relu1 = nn.ReLU()  # Hàm kích hoạt ReLU
        self.layer2 = nn.Linear(512, 256)  # Lớp fully connected thứ hai
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 10)  # Lớp đầu ra (10 lớp cho 0-9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.flatten = nn.Flatten()
        
        # BatchNorm và Wide layer đầu tiên
        self.bn1 = nn.BatchNorm1d(784)  # BatchNorm cho input 28*28
        self.wide1 = nn.Linear(784, 1024)  # Wide layer
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        
        # Deep path
        self.deep1 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.deep2 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Output layer
        self.output = nn.Linear(256 + 1024, 10)  # Kết hợp deep và wide
        
    def forward(self, x):
        # Flatten input
        x = self.flatten(x)  # [batch_size, 784]
        
        # BatchNorm
        x = self.bn1(x)
        
        # Wide path
        wide = self.wide1(x)  # [batch_size, 1024]
        wide = F.relu(wide)
        
        # Attention - cần thêm chiều sequence (tạm dùng 1)
        attn_input = wide.unsqueeze(0)  # [1, batch_size, 1024]
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.squeeze(0)  # [batch_size, 1024]
        
        # Deep path
        deep = self.deep1(attn_output)  # [batch_size, 512]
        deep = self.bn2(deep)
        deep = F.relu(deep)
        deep = self.deep2(deep)  # [batch_size, 256]
        deep = self.bn3(deep)
        deep = F.relu(deep)
        
        # Kết hợp Deep và Wide
        combined = torch.cat([deep, wide], dim=1)  # [batch_size, 256 + 1024]
        
        # Output
        out = self.output(combined)
        return out

# 5. Huấn luyện và đánh giá mô hình
def train(model, model_path, train_loader, val_loader, criterion, optimizer, epochs=5):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}] Train', leave=False)
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # Cập nhật progress bar
            train_loop.set_postfix({'Train Loss': train_loss / (train_loop.n + 1)})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{epochs}] Val', leave=False)
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Cập nhật progress bar
                val_loop.set_postfix({'Val Loss': val_loss / (val_loop.n + 1)})

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # In kết quả
        print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Lưu model nếu val_accuracy cao hơn
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f'Saved model with Val Accuracy: {best_val_acc:.2f}%')

if __name__ == '__main__':
    # 4. Khởi tạo mô hình, loss function và optimizer
    # model = SimpleNN().to(device)
    model = EnhancedNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    root = os.getcwd()
    model_path = os.path.join(root, "weights", "enhanced.pth")
    train(model, model_path, train_loader, val_loader, criterion, optimizer, epochs=5)