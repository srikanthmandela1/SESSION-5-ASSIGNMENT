#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
import torch.nn.functional as F

class NetWithGroupNorm(nn.Module):
    def __init__(self):
        super(NetWithGroupNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(2, 6) # 2 groups with 6 channels each
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(4, 16) # 4 groups with 16 channels each
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[2]:


class NetWithLayerNorm(nn.Module):
    def __init__(self):
        super(NetWithLayerNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.ln1 = nn.LayerNorm([6, 28, 28]) # apply layer norm after first conv
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ln2 = nn.LayerNorm([16, 10, 10]) # apply layer norm after second conv
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.ln1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.ln2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[3]:


class NetWithL1andBN(nn.Module):
    def __init__(self):
        super(NetWithL1andBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def l1_regularization(self, l1_lambda):
        l1 = 0
        for p in self.parameters():
            l1 = l1 + p.abs().sum()
        return l1 * l1_lambda


# In[ ]:




