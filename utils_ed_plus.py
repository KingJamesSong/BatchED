import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torch.autograd import Function
from mpmath import *
import math
import numpy as np
eps = 1e-10
mp.dps = 20
interval_count=4
constant_one = torch.Tensor([1.])
one = mpf(1)
mp.pretty = True
def f(x):
    return one/(one-x)
a = taylor(f, 0, 100)
#[50, 50] Pade coefficients of the geometric series
pade_p, pade_q = pade(a, 50, 50)
pade_p = torch.from_numpy(np.array(pade_p).astype(float)).cuda()
pade_q = torch.from_numpy(np.array(pade_q).astype(float)).cuda()

def pade_approximants(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])#.transpose(1,2)
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)*pade_p[0]
    q_app = torch.ones_like(p)*pade_q[0]
    p_hat = torch.ones_like(p)
    for i in range(50):
        p_hat = p_hat * p
        p_app += pade_p[i+1]*p_hat
        q_app += pade_q[i+1]*p_hat
    a1 = a1 * p_app / q_app #rational approximation
    return a1


def taylor_polynomial(s):
    s = torch.diagonal(s, dim1=1, dim2=2)
    dtype = s.dtype
    I = torch.eye(s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p)
    a1 = s.view(s.shape[0],s.shape[1],1).repeat(1, 1, s.shape[1])
    a1_t = a1.transpose(1,2)
    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    a1 *= torch.ones(s.shape[1], s.shape[1], device=s.device).type(dtype).view(1,s.shape[1],s.shape[1]).repeat(s.shape[0],1,1) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(100):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = a1 * p_app
    return a1

class Batched_ED_Plus(Function):
    @staticmethod
    def forward(ctx, input):
        cov = input
        batch_size, input_dim,_ = input.size()
        # Householder reflector to tri-diagonal form
        tri_cov, tri_vector = Householder_reflection(cov,batch_size,input_dim)
        # Divide Step
        break_times = int(math.log2(input_dim))
        norm_list=[]
        beta_list=[]
        for i in range(break_times-1):
            tri_smaller, covnorm, cov_beta = DivideStep(tri_cov)
            norm_list.append(covnorm)
            beta_list.append(cov_beta)
            tri_cov = tri_smaller
        # QR iteration to compute smallest block
        conquer_index = break_times-2
        #_, fused_diag, fused_vector = torch.svd(tri_smaller)
        #fused_diag = fused_diag.diag_embed()
        #fused_diag, fused_vector = QR_Single(tri_smaller)
        fused_diag, fused_vector = QR_Givens_Shift(tri_smaller,tri_smaller.size(0),tri_smaller.size(1),tri_smaller.size(1))
        #print(fused_diag)
        #print(fused_diag1)
        #fused_diag = torch.diag_embed(torch.clamp(torch.diagonal(fused_diag, dim1=1, dim2=2), min=1e-10))
        #Conquer step takes care of the rest
        while conquer_index>=0:
            fused_vector, fused_diag = ConquerStep(norm_list[conquer_index], beta_list[conquer_index], fused_vector, fused_diag)
            #fused_diag = torch.diag_embed(torch.clamp(torch.diagonal(fused_diag, dim1=1, dim2=2), min=1e-10))
            conquer_index = conquer_index - 1
        eig_diag = torch.diag_embed(torch.clamp(torch.diagonal(fused_diag,dim1=1,dim2=2),min=1e-10))
        #eig_diag = fused_diag
        eig_vec = tri_vector.bmm(fused_vector)
        ctx.save_for_backward(eig_vec, eig_diag)
        return eig_vec,eig_diag

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        eig_vec, eig_diag = ctx.saved_tensors
        eig_vec_deri, eig_diag_deri = grad_output1, grad_output2
        k = taylor_polynomial(eig_diag)
        # Gradient Overflow Check
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        grad_input = (k.transpose(1, 2) * (eig_vec.transpose(1, 2).bmm(eig_vec_deri))) + torch.diag_embed(
            torch.diagonal(eig_diag_deri, dim1=1, dim2=2))
        # Gradient Overflow Check
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        grad_input = eig_vec.bmm(grad_input).bmm(eig_vec.transpose(1, 2))
        # Gradient Overflow Check
        grad_input[grad_input == float('inf')] = grad_input[grad_input != float('inf')].max()
        grad_input[grad_input == float('-inf')] = grad_input[grad_input != float('-inf')].min()
        return grad_input

def wilkinson_shift(matrix):
  sigma = (matrix[:,0,0]-matrix[:,1,1])/2
  temp = (torch.sign(sigma) * matrix[:,0,1]**2) / (abs(sigma)+torch.sqrt(sigma**2+matrix[:,0,1]**2)+eps)
  return matrix[:,0,0]-temp, matrix[:,1,1] - temp

def house_two_sides(A,mu):
  H = 0.5 * (torch.norm(mu+eps,dim=1).type(A.dtype).view(mu.size(0),1,1))**2
  p = A.bmm(mu)/H
  K = mu.transpose(1,2).bmm(p)/2/H
  q = p - K*(mu)
  return A - q.bmm(mu.transpose(1,2)) - mu.bmm(q.transpose(1,2))

def QR_Givens_Shift(cov,batch_size,input_dim,iterations):
  Givens = torch.eye(input_dim,device=cov.device).view(1,input_dim,input_dim,1).repeat(batch_size,1,1,input_dim-1)
  shift = torch.eye(input_dim,device=cov.device).view(1,input_dim,input_dim).repeat(batch_size,1,1)
  reduction=0
  Givens_single = shift.clone()
  Givens_all = shift.clone()
  for iteration in range(iterations):
    shift_coe1, shift_coef2  = wilkinson_shift(cov[:,input_dim-2-reduction:input_dim-reduction,input_dim-2-reduction:input_dim-reduction])
    ## 1st Shift
    # 1st Rotation
    tmp_shift = shift_coef2.view(shift.size(0),1,1)*shift
    cov  =  cov - tmp_shift
    Givens[:,0:2,0,0] = cov[:,0,0:2] / (torch.sqrt(torch.sum(cov[:,0,0:2]**2,dim=1)).view(cov.size(0),1)+eps)
    Givens[:, 0, 0, 0] = torch.where(Givens[:,0,0,0]**2 + Givens[:,0,1,0]**2 == 0, constant_one.to(Givens), Givens[:,0,0,0]) # In case the matrix is all-zero
    Givens[:,1,1,0] = Givens[:,0,0,0]
    Givens[:,0,1,0] = - Givens[:,1,0,0]
    cov[:,0:3,0:3]=Givens[:,0:3,0:3,0].transpose(1, 2).bmm(cov[:,0:3,0:3]).bmm(Givens[:,0:3,0:3,0])
    Givens_single[:,0:3,0:3] = Givens_single[:,0:3,0:3].bmm(Givens[:,0:3,0:3,0])
    for i in range(1,input_dim-1-reduction):
      Givens[:,i:i+2,i,i] = cov[:,i,i:i+2] / (torch.sqrt(torch.sum(cov[:,i,i:i+2]**2,dim=1)).view(cov.size(0),1)+eps)
      Givens[:,i+1,i+1,i] = Givens[:,i,i,i]
      Givens[:,i,i+1,i] = - Givens[:,i+1,i,i]
      cov[:,i-1:i+3,i-1:i+3] = Givens[:,i-1:i+3,i-1:i+3,i].transpose(1, 2).bmm(cov[:,i-1:i+3,i-1:i+3]).bmm(Givens[:,i-1:i+3,i-1:i+3,i])
      Givens_single[:,:,i:] = Givens_single[:,:,i:].bmm(Givens[:,i:,i:,i])
    Givens_all = Givens_all.bmm(Givens_single)
    Givens_single = shift.clone()
    cov = cov + tmp_shift
    if abs(cov[:,input_dim-1-reduction,input_dim-2-reduction]).max()<1e-5:
      reduction = reduction + 1
    if reduction>=input_dim-2:
      break
    ### 2nd Shift
    tmp_shift = shift_coe1.view(shift.size(0),1,1)*shift
    cov  =  cov - tmp_shift
    Givens[:,0:2,0,0] = cov[:,0,0:2] / (torch.sqrt(torch.sum(cov[:,0,0:2]**2,dim=1)).view(cov.size(0),1)+eps)
    Givens[:, 0, 0, 0] = torch.where(Givens[:,0,0,0]**2 + Givens[:,0,1,0]**2 == 0, constant_one.to(Givens), Givens[:,0,0,0])
    Givens[:,1,1,0] = Givens[:,0,0,0]
    Givens[:,0,1,0] = - Givens[:,1,0,0]
    cov[:,0:3,0:3]=Givens[:,0:3,0:3,0].transpose(1, 2).bmm(cov[:,0:3,0:3]).bmm(Givens[:,0:3,0:3,0])
    Givens_single[:,0:3,0:3] = Givens_single[:,0:3,0:3].bmm(Givens[:,0:3,0:3,0])
    for i in range(1,input_dim-1-reduction):
      Givens[:,i:i+2,i,i] = cov[:,i,i:i+2] / (torch.sqrt(torch.sum(cov[:,i,i:i+2]**2,dim=1)).view(cov.size(0),1)+eps)
      Givens[:,i+1,i+1,i] = Givens[:,i,i,i]
      Givens[:,i,i+1,i] = - Givens[:,i+1,i,i]
      cov[:,i-1:i+3,i-1:i+3] = Givens[:,i-1:i+3,i-1:i+3,i].transpose(1, 2).bmm(cov[:,i-1:i+3,i-1:i+3]).bmm(Givens[:,i-1:i+3,i-1:i+3,i])
      Givens_single[:,:,i:] = Givens_single[:,:,i:].bmm(Givens[:,i:,i:,i])
    Givens_all = Givens_all.bmm(Givens_single)
    Givens_single = shift.clone()
    cov = cov + tmp_shift
    if abs(cov[:,input_dim-1-reduction,input_dim-2-reduction]).max()<1e-5:
      reduction = reduction + 1
    if reduction>=input_dim-2:
      break
  return cov, Givens_all

def Householder_reflection(cov, batch_size, input_dim):
    hv = torch.zeros(batch_size, input_dim, input_dim - 2, requires_grad=False, device=cov.device)
    eigen_vectors = torch.eye(input_dim, requires_grad=False, device=cov.device).view(1, input_dim, input_dim).repeat(
        batch_size, 1, 1)
    eye_matrix = torch.eye(input_dim, requires_grad=False, device=cov.device).view(1, input_dim, input_dim).repeat(
        batch_size, 1, 1)
    for i in range(2, input_dim):
        cov_temp = cov.clone()
        hv[:, i - 1:input_dim, i - 2] = cov_temp[:, i - 2, i - 1:input_dim]
        hv[:, i - 1, i - 2] = hv[:, i - 1, i - 2] + (hv[:, i - 1, i - 2] / (abs(hv[:, i - 1, i - 2])+eps))* torch.sqrt(
            torch.sum(hv[:, i - 1:input_dim, i - 2] ** 2, dim=1))
        hm = eye_matrix - 2 * torch.bmm(hv[:, :, i - 2:i - 1], (hv[:, :, i - 2:i - 1].transpose(1, 2))) / (
            torch.norm(hv[:, :, i - 2]+eps, dim=1).view(batch_size, 1, 1)) ** 2
        eigen_vectors = eigen_vectors.bmm(hm)
        cov = house_two_sides(cov, hv[:, :, i - 2:i - 1])
    return cov, eigen_vectors

def frobenius_norm(x):
    return torch.sqrt(torch.sum(x**2,dim=[1,2]))


def DivideStep(x_orig):
    x = x_orig
    xnorm = frobenius_norm(x_orig).unsqueeze(dim=1)
    half_size = int(x.size(1) / 2)
    beta = x[:, half_size, half_size - 1]
    betaabs = beta.abs()
    x[:, half_size - 1, half_size - 1] = x[:, half_size - 1, half_size - 1] - betaabs
    x[:, half_size, half_size] = x[:, half_size, half_size] - betaabs
    t1 = x[:, :half_size, :half_size]
    t2 = x[:, half_size:, half_size:]
    return torch.cat([t1,t2],dim=0), xnorm, beta

def ConquerStep(xnorm, beta , t_vec, t_eig):
    #Divide two T in batch dimension
    B = int(t_vec.size(0)/2)
    half_size = t_vec.size(1)
    full_size = half_size * 2
    betaabs = beta.abs()
    t1_vec = t_vec[:B,:,:]
    t2_vec = t_vec[B:, :, :]
    t1_diag = t_eig[:B,:,:]
    t2_diag = t_eig[B:, :, :]

    d_diag = torch.zeros(full_size, full_size, device=t_vec.device, requires_grad=False).view(1, full_size,full_size).repeat(B,1,1)
    q_vec = torch.zeros(full_size, full_size, device=t_vec.device, requires_grad=False).view(1, full_size,full_size).repeat(B,1,1)
    z_diag = torch.zeros(full_size, 1, device=t_vec.device, requires_grad=False).view(1, full_size, 1).repeat(B, 1, 1)

    d_diag[:, :half_size, :half_size] = t1_diag
    d_diag[:, half_size:, half_size:] = t2_diag

    q_vec[:, :half_size, :half_size] = t1_vec
    q_vec[:, half_size:, half_size:] = t2_vec
    z_diag[:, :half_size, :] = torch.sign(beta).unsqueeze(dim=1).unsqueeze(dim=1) * t1_vec.transpose(1, 2)[:, :, half_size - 1:half_size]
    z_diag[:, half_size:, :] = t2_vec.transpose(1, 2)[:, :, 0:1]
    def secular_nonbatch(t_diag_, beta_=None, z_diag_=None, d_diag_=None,grad_require=False):
        sum = 1
        grad = 0
        grad_grad = 0
        if grad_require == True:
            for i in range(full_size):
                temp_no = beta_.unsqueeze(dim=1) *(z_diag_[:, i:i+1, 0] ** 2)
                temp_de = (d_diag_[:, i:i+1] - t_diag_+eps)
                div = temp_no / temp_de
                sum  +=  div
                div_div = div / (temp_de)
                grad += div_div
                grad_grad += 2 * div_div / (temp_de)
        else:
            for i in range(full_size):
                sum +=  betaabs.unsqueeze(dim=1) *(z_diag[:, i:i+1, 0] ** 2) / (d_diag_2d[:, i:i+1] - t_diag+eps)
        return  sum, grad, grad_grad

    def HalleyRootFind(g, z_0, eps, lower_interval, upper_interval, max_iters):
        prevZ = z_0
        prevG, prev_grad, prev_grad_grad =g(prevZ,betaabs,z_diag,d_diag_2d,grad_require=True)
        iters = 0
        delta = torch.ones_like(beta)
        tol = delta.max()
        while eps < tol and iters < max_iters:
            batch_index= (delta > eps).nonzero().squeeze(dim=1)
            if len(batch_index.size())==0:
                break
            iters += 1
            prev_prevZ = prevZ.clone()
            z_temp = prevZ[batch_index]
            z_temp = z_temp - (2 * prevG[batch_index] * prev_grad[batch_index]) / (2 * (prev_grad[batch_index]**2) - prevG[batch_index] * prev_grad_grad[batch_index])
            z_temp = torch.where(z_temp < lower_interval[batch_index], (prev_prevZ[batch_index] + lower_interval[batch_index]) / 2, z_temp)
            z_temp = torch.where(z_temp > upper_interval[batch_index], (prev_prevZ[batch_index] + upper_interval[batch_index]) / 2, z_temp)
            prevZ[batch_index] = z_temp
            temp1, temp2, temp3 = g(prevZ[batch_index],betaabs[batch_index],z_diag[batch_index],d_diag_2d[batch_index],grad_require=True)
            prevG[batch_index], prev_grad[batch_index], prev_grad_grad[batch_index] = temp1, temp2, temp3
            delta = torch.norm(prevZ-prev_prevZ,dim=[1])
            tol = delta.max()
        return prevZ, iters
    # Define Closed Interval for eigenvalues
    d_diag_2d = torch.diagonal(d_diag, dim1=1, dim2=2)
    t_diag = torch.zeros(B,d_diag_2d.size(1),device=d_diag_2d.device)
    lower_interval = d_diag_2d.clone()
    sorted_diag, index_diag = torch.sort(d_diag_2d, dim=1, descending=True)
    sorted_diag = torch.cat([xnorm, sorted_diag], dim=1)
    upper_interval = torch.gather(sorted_diag, dim=1, index=index_diag.argsort(1))
    # Fist Multi-section step
    t_diag = lower_interval + (upper_interval- lower_interval) / interval_count
    # Perform Hybrid Multi-section and Bi-section Methods
    i = 0
    tol = float('inf')
    while i < 20 and tol>1e-2:
        secu_diag, _, _= secular_nonbatch(t_diag, grad_require=False)
        lower_interval = torch.where(secu_diag > 0, lower_interval, t_diag)
        upper_interval = torch.where(secu_diag < 0, upper_interval, t_diag)
        if i%2 ==0:
            t_diag = (lower_interval+upper_interval)/2 #lower_interval + (upper_interval - lower_interval) / interval_count/3
        else:
            t_diag= lower_interval + (upper_interval - lower_interval) / interval_count
        tol = torch.norm(upper_interval-lower_interval)
        i=i+1
    #Newton' method or Halley's method for root-finding
    t_diag, iter = HalleyRootFind(secular_nonbatch, t_diag, 1e-5, lower_interval, upper_interval, 20)
    #print(i,iter)
    t_vec_D = torch.zeros(full_size, full_size, device=t_diag.device).view(1, full_size, full_size).repeat(B, 1, 1)
    #Gu's backward-stable method to compute eigenvector
    new_z = torch.zeros_like(z_diag).squeeze(dim=2)
    new_z[:, 0] = torch.prod((t_diag[:, 1:] - d_diag_2d[:, 0:1]+eps) / (d_diag_2d[:, 1:] - d_diag_2d[:, 0:1]+eps),
                             dim=1) * (t_diag[:, 0] - d_diag_2d[:, 0]+eps)
    for i in range(1, full_size - 1):
        new_z[:, i] = torch.prod(torch.cat(
            [(t_diag[:, :i] - d_diag_2d[:, i:i + 1]+eps ) / (d_diag_2d[:, :i] - d_diag_2d[:, i:i + 1]+eps ),
             (t_diag[:, i + 1:] - d_diag_2d[:, i:i + 1] +eps) / (d_diag_2d[:, i + 1:] - d_diag_2d[:, i:i + 1]+eps )],
            dim=1),
            dim=1) * (t_diag[:, i] - d_diag_2d[:, i]+eps)
    new_z[:, full_size - 1] = torch.prod((t_diag[:, :full_size - 1] - d_diag_2d[:, full_size - 1:full_size]+eps ) / (
            d_diag_2d[:, :full_size - 1] - d_diag_2d[:, full_size - 1:full_size]+eps ), dim=1) * (
                                      t_diag[:, full_size - 1] - d_diag_2d[:, full_size - 1]+eps)
    new_z = torch.sqrt(torch.abs(new_z) / betaabs.view(B, 1))
    if torch.isnan(new_z).any():
        new_z = z_diag.squeeze()
    for i in range(full_size):
        t_vec_D[:,:,i] = torch.sign(z_diag).squeeze()*new_z / (d_diag_2d-t_diag[:,i:i+1]+eps)
        t_vec_D[:,:,i] = t_vec_D[:,:,i] / (torch.norm(t_vec_D[:,:,i],dim=1,keepdim=True))
    eigenvec = q_vec.bmm(t_vec_D)
    return eigenvec, torch.diag_embed(t_diag)