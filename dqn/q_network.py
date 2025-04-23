import torch.nn as nn
import torch.nn.functional as F

class QNetworkShallow(nn.Module):
    """Simple network: 8 → 64 → 5"""
    def __init__(self, state_size=8, action_size=5):
        super(QNetworkShallow, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)    # Input (8) → Hidden (64)
        self.fc2 = nn.Linear(64, action_size)   # Hidden (64) → Output (5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class QNetworkMedium(nn.Module):
    """Medium network: 8 → 128 → 64 → 5 (with dropout)"""
    def __init__(self, state_size=8, action_size=5):
        super(QNetworkMedium, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)   # Input (8) → Hidden1 (128)
        self.fc2 = nn.Linear(128, 64)           # Hidden1 (128) → Hidden2 (64)
        self.fc3 = nn.Linear(64, action_size)   # Hidden2 (64) → Output (5)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class QNetworkDeep(nn.Module):
    """Deep network: 8 → 256 → 128 → 64 → 5"""
    def __init__(self, state_size=8, action_size=5):
        super(QNetworkDeep, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)   # Input (8) → Hidden1 (256)
        self.fc2 = nn.Linear(256, 128)          # Hidden1 (256) → Hidden2 (128)
        self.fc3 = nn.Linear(128, 64)           # Hidden2 (128) → Hidden3 (64)
        self.fc4 = nn.Linear(64, action_size)   # Hidden3 (64) → Output (5)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second layer
        x = F.relu(self.fc3(x))
        return self.fc4(x)
    
class QNetworkTief(nn.Module):
    """Deep network: 8 → 512 → 256 → 128 → 64 → 5"""
    def __init__(self, state_size=8, action_size=5):
        super(QNetworkDeep, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)   # Input (8) → Hidden1 (512)
        self.fc2 = nn.Linear(512, 256)          # Input (512) → Hidden1 (256)
        self.fc3 = nn.Linear(256, 128)          # Hidden1 (256) → Hidden2 (128)
        self.fc4 = nn.Linear(128, 64)           # Hidden2 (128) → Hidden3 (64)
        self.fc5 = nn.Linear(64, action_size)   # Hidden3 (64) → Output (5)
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second layer
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after third layer
        x = F.relu(self.fc4(x))
        return self.fc5(x) 