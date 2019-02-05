import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	
	def __init__(self, latent_dim=2):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 500)
		self.fc2 = nn.Linear(500, 500)
		self.fc3 = nn.Linear(500, latent_dim)
	
		for m in self.modules():
			if type(m) == nn.Linear:
				nn.init.kaiming_normal_(m.weight)
				m.bias.data.fill_(0.)
	
	def forward(self, x):
		x = x.view(-1, 28*28)
		
		x = self.fc1(x)
		x = F.elu(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc2(x)
		x = torch.tanh(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc3(x)
		return x

class Decoder(nn.Module):
	
	def __init__(self, latent_dim=2):
		super().__init__()
		self.fc1 = nn.Linear(latent_dim, 500)
		self.fc2 = nn.Linear(500, 500)
		self.fc3 = nn.Linear(500, 28*28)
		
		for m in self.modules():
			if type(m) == nn.Linear:
				nn.init.kaiming_normal_(m.weight)
				m.bias.data.fill_(0.)
	
	def forward(self, x):
		x = self.fc1(x)
		x = torch.tanh(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc2(x)
		x = F.elu(x)
		x = F.dropout(x, p=0.1)
		
		x = self.fc3(x)
		x = torch.sigmoid(x)
		
		x = x.view(-1, 1, 28, 28)
		return x

class Autoencoder(nn.Module):
	
	def __init__(self, latent_dim=2):
		super().__init__()
		self.encoder = Encoder(latent_dim)
		self.decoder = Decoder(latent_dim)
	
	# x is in range [0, 1]
	def forward(self, x):
		return self.decoder(self.encoder(x))