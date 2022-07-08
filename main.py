from utils_ed import Batched_ED
import torch
import time

batched_ed = Batched_ED.apply
batch_size = 512
input_dim = 4

x=torch.randn(batch_size,input_dim,input_dim,requires_grad=False).cuda()
cov = x.bmm(x.transpose(1,2))
torch.cuda.synchronize()
t0 = time.time()
_,eig_diag,eig_vec = torch.svd(cov)
t1= time.time()
torch.cuda.synchronize()
print("SVD Time", t1-t0)

torch.cuda.synchronize()
t0 = time.time()
eigen_vectors,eigen_values = batched_ed(cov)
t1= time.time()
torch.cuda.synchronize()
print("Batched ED Time", t1-t0)

