import torchfile
import torch


if __name__ == '__main__':
	# input argument(s)
	t7_model = 'PIV-LiteFlowNet-en.t7'
	pytorch_model = 'PIV-LiteFlowNet-en.pth'
	test = 'PIV_LiteFlowNet_en.pth'

	# instantiate the model in pytorch
	model = torchfile.load(t7_model)
	model_dir = model

	test_model = torch.load(test)

	# save model in pytorch format
	torch.save(model, pytorch_model)

	print('DONE!')
