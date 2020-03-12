require 'loadcaffe'


-- import prototxt and caffemodel
prototxt_file = 'PIV-LiteFlowNet-en_deploy.prototxt'
caffemodel_file = 'PIV-LiteFlowNet-en.caffemodel'

-- output file
save_model = './PIV-LiteFlowNet-en.t7'

-- instantiate the model
model = loadcaffe.load(prototxt_file, caffemodel_file, 'nn')

-- saving model
torch.save(save_model, model)