import torch
import torch.nn as nn

class OurModule(nn.Module):
	"""docstring for OurModule"""
	def __init__(self, num_inputs, num_classes, dropout_prob=0.3):
		super(OurModule, self).__init__()
		self.pipe = nn.Sequential(
			nn.Linear(num_inputs, 5), 
			nn.ReLU(),
			nn.Linear(5, 20), 
			nn.ReLU(),
			nn.Linear(20, num_classes), 
			nn.Dropout(p=dropout_prob),
			nn.Softmax()
		)

	def forward(self, x):
		return self.pipe(x)

if __name__ == '__main__':
	net = OurModule(2,3,0.3)
	v = torch.FloatTensor([[2, 3]])
	out = net(v)
	print(net)
	print(out)

# the following is pseudo code from page 64 ...
def train(data):
	for batch_samples, batch_labels in iterate_batches(data, batch_size=32):
		batch_samples_t = torch.tensor(batch_samples)
		batch_labels_t  = torch.tensor(batch_labels)
		out_t = net(batch_samples_t)
		loss_t = loss_function(out_t, batch_labels_t)
		loss_t.backward()
		optimizer.step()
		optimizer.zero_grad()