import torch
import torch.nn.functional as nfunc
import numpy as np

def _l2_normalize(d):
    t = d.clone()   # remove from the computing graph
    norm = torch.sqrt(torch.sum(t.view(d.shape[0], -1) ** 2, dim=1))
    if len(t.shape) == 4:
        norm = norm.reshape(-1, 1, 1, 1)
    elif len(t.shape) == 3:
        norm = norm.reshape(-1, 1, 1)
    elif len(t.shape) == 2:
        norm = norm.reshape(-1, 1)
    else:
        raise NotImplementedError
    normed_d = t / (norm + 1e-10)
    return normed_d




class VAT(object):

    def __init__(self, device, eps, xi, k=1):
        self.device = device
        self.xi = xi
        self.eps = eps
        self.k = k

    def __call__(self, model, image):
        logits = model(image)
        prob_x = nfunc.softmax(logits.detach(), dim=1)
        # np generator is more controllable than torch.randn(image.size())
        d = np.random.standard_normal(image.size())
        d = _l2_normalize(torch.FloatTensor(d).to(self.device))

        for ip in range(self.k):
            d *= self.xi
            d.requires_grad = True
            t = image.detach()
            x_hat = t + d
            logits_x_hat = model(x_hat)
            
            # official theano code compute in this way
            # log_prob_x_hat = torch.log(logits_x_hat)
            log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
            adv_distance = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
            adv_distance.backward()
            grad_x_hat = d.grad
            d = _l2_normalize(grad_x_hat).to(self.device)

        logits_x_hat = model(image + self.eps * d)
        # official theano code works in this way
        log_prob_x_hat = nfunc.log_softmax(logits_x_hat, dim=1)
        lds = torch.mean(torch.sum(- prob_x * log_prob_x_hat, dim=1))
        return lds
