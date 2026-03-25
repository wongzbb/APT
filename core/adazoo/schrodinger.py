import copy
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from core.param import load_model_and_optimizer, copy_model_and_optimizer
import time
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import os


class SchrodingerModel(nn.Module):
    def __init__(self, model, n_classes):
        super(SchrodingerModel, self).__init__()
        self.f = model
        if n_classes < 200:
            self.schrodinger_head = nn.Sequential(
                nn.Linear(1, 128),  nn.SiLU(),  #64->128
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(),
                nn.Linear(128, 2), 
            ).to(next(model.parameters()).device)
        else:
            self.schrodinger_head = nn.Sequential(
                nn.Linear(1, 256),  nn.SiLU(),  #64->128
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, 2), 
            ).to(next(model.parameters()).device)
        # self.register_buffer('a', torch.tensor(-1000.0, dtype=torch.float32, requires_grad=False))
        self.a = nn.Parameter(torch.tensor(-1000.0), requires_grad=False).to(next(model.parameters()).device)
        # self.a = torch.tensor(-1000.0, dtype=torch.float32, device=next(model.parameters()).device, requires_grad=False)

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        # if y is None:
        L_x = -torch.logsumexp(logits, 1, keepdim=True)       
        psi = self.schrodinger_head(L_x)

        return logits, L_x, psi        

            
class Schrodinger(nn.Module):
    def __init__(self, model, optimizer, steps=1, threshold=0.01, lr=0.01, n_classes=10, im_sz=32, n_ch=3, path=None, logger=None, episodic=False, epsilon=1e-3, alpha=1.0, beta=1.0, gamma=1.0, h=1.0, m=1.0, E=1.0, V0=1.0, head_lr=0.001, boundary_left=-100.0, boundary_right=0.0, delta=1.0, ada_head_during_test=False, InD_threshold=0.9, use_schrodinger_loss=True, zeta=1.0):
        super().__init__()
        self.schrodinger_model = SchrodingerModel(model, n_classes)
        self.optimizer = optimizer
        # self.optimizer.add_param_group({"params": self.schrodinger_model.schrodinger_head.parameters()})  #important
        self.optimizer_head = torch.optim.Adam(self.schrodinger_model.schrodinger_head.parameters(), lr=head_lr)
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.lr = lr
        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        self.path = path
        self.logger = logger   
        self.episodic = episodic
        self.threshold = threshold
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.h = h
        self.m = m
        self.E = E
        self.V0 = V0
        self.boundary_left = boundary_left
        self.boundary_right = boundary_right
        self.delta = delta
        self.ada_head_during_test = ada_head_during_test
        self.InD_threshold = InD_threshold
        self.use_schrodinger_loss = use_schrodinger_loss
        self.zeta = zeta
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.schrodinger_model.f, self.optimizer)
        # self.load_head_path = load_head_path

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if if_adapt:
            if self.episodic:
                self.reset()
            with alive_bar(self.steps, title="Processing") as bar:
                for _ in range(self.steps):
                    time.sleep(0.1)
                    bar()
                    classifier_output, _, _ = self.forward_and_adapt(x, self.schrodinger_model, self.optimizer, self.ada_head_during_test)
                    # if i % 1 == 0 and if_vis:
                    #     self.visualize_images(path=self.path, schrodinger_model=self.schrodinger_model, 
                    #                     batch_size=100, n_classes=self.n_classes, im_sz=self.im_sz, n_ch=self.n_ch, device=x.device, counter=counter, step=i)
        else:
            self.schrodinger_model.eval()
            with torch.no_grad():
                classifier_output, _, _ = self.schrodinger_model(x)

        return classifier_output

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer, ada_head_during_test):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        BATCH_SIZE = x.shape[0]
        x = x.requires_grad_(True)

        classifier_output, x, psi = model(x)
        sum_squares = (psi ** 2).sum(dim=-1)
        x_ = x.squeeze()
        # print(torch.max(x_))
        mask = x_ < self.schrodinger_model.a.data
        InD_output = classifier_output[mask] 
        OD_sum_squares = sum_squares[~mask]
        print(InD_output.shape[0], OD_sum_squares.shape[0])
        # print('------------------')

        total_loss = torch.tensor(0.0).to(x.device)
        # if InD_output.shape[0] != 0:
        #     py, y_prime = F.softmax(InD_output, dim=-1).max(1)
        #     flag = py > self.InD_threshold
        #     ind_loss = F.cross_entropy(InD_output[flag], y_prime[flag])
        # else:
        #     ind_loss = torch.tensor(0.0).to(x.device)
        # total_loss += ind_loss/ind_loss.item() * self.alpha
        # self.logger.info(f"ind_loss: {ind_loss.item()}")

        ood_loss = torch.tensor(0.0).to(x.device)
        if OD_sum_squares.shape[0] != 0:
            ood_loss = torch.mean(OD_sum_squares)
            total_loss += ood_loss/ood_loss.item() * self.beta
        if self.use_schrodinger_loss:
            # print('schrodinger loss')
            grad_psi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
            grad_grad_psi = torch.autograd.grad(grad_psi, x, grad_outputs=torch.ones_like(grad_psi), create_graph=True, retain_graph=True)[0]
            mask_left = (x < self.schrodinger_model.a).float()
            mask_right = (x >= self.schrodinger_model.a).float()
            ###########
            residual_left = (-(self.h**2/(2*self.m))) * grad_grad_psi - self.E * psi
            residual_right = (-(self.h**2/(2*self.m))) * grad_grad_psi + self.V0 * psi - self.E * psi
            pde_loss = (mask_left * torch.abs(residual_left)**2 + mask_right * torch.abs(residual_right)**2).mean()
            ###########
            x_left = torch.full((x.shape[0],1), self.schrodinger_model.a-self.epsilon, requires_grad=True).to(x.device)
            x_right = torch.full((x.shape[0],1), self.schrodinger_model.a+self.epsilon, requires_grad=True).to(x.device)
            psi_left = self.schrodinger_model.schrodinger_head(x_left)
            psi_right = self.schrodinger_model.schrodinger_head(x_right)
            bc_value_loss = torch.abs(psi_left - psi_right).pow(2).mean()
            ###########
            grad_left = torch.autograd.grad(psi_left.sum(), x_left, create_graph=True)[0]
            grad_right = torch.autograd.grad(psi_right.sum(), x_right, create_graph=True)[0]
            bc_grad_loss = torch.abs(grad_left - grad_right).pow(2).mean()
            ###########
            x_boundary_left = torch.full((x.shape[0]//2,1), self.boundary_left).to(x.device)
            x_boundary_right = torch.full((x.shape[0]//2, 1), self.boundary_right).to(x.device)
            x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0).requires_grad_(True)
            psi_boundary = self.schrodinger_model.schrodinger_head(x_boundary)
            real_b, imag_b = psi_boundary[:, 0:1], psi_boundary[:, 1:2]
            bc_domain_loss = (real_b.pow(2).mean() + imag_b.pow(2).mean())

            self.logger.info(f"ood_loss: {ood_loss.item()}, pde_loss: {pde_loss.item()}, bc_value_loss: {bc_value_loss.item()}, bc_grad_loss: {bc_grad_loss.item()}, bc_domain_loss: {bc_domain_loss.item()}")

            if pde_loss.item() != 0:
                pde_loss = pde_loss/pde_loss.item()
            if bc_value_loss.item() != 0:
                bc_value_loss = bc_value_loss/bc_value_loss.item()
            if bc_grad_loss.item() != 0:
                bc_grad_loss = bc_grad_loss/bc_grad_loss.item()
            if bc_domain_loss.item() != 0:
                bc_domain_loss = bc_domain_loss/bc_domain_loss.item()

            total_loss += (pde_loss + self.gamma* (bc_value_loss + bc_grad_loss) + self.delta*bc_domain_loss) * self.zeta
            
                
        self.logger.info(f"total_loss: {total_loss.item()}")

        
        # optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return classifier_output, x, psi


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.schrodinger_model.f, self.optimizer,
                                 self.model_state, self.optimizer_state)
        
    def set_head_optimizer(self):
        self.schrodinger_model.train()
        self.schrodinger_model.f.requires_grad_(False)
        self.schrodinger_model.schrodinger_head.requires_grad_(True)

    def reset_schrodinger_head(self, head_path, ada_head_during_test):
        self.schrodinger_model.train()
        self.schrodinger_model.f.requires_grad_(True)

        checkpoint = torch.load(head_path)
        self.schrodinger_model.schrodinger_head.load_state_dict(checkpoint['model_state_dict'])
        print(checkpoint['a'])
        try:
            self.schrodinger_model.a.copy_(torch.tensor(checkpoint['a'][0]))
            self.boundary_left = checkpoint['a'][1]
        except:
            self.schrodinger_model.a = torch.tensor(checkpoint['a'])

        if ada_head_during_test:
            self.schrodinger_model.schrodinger_head.requires_grad_(True)
            self.optimizer.add_param_group({"params": self.schrodinger_model.schrodinger_head.parameters()})
        else:
            self.schrodinger_model.schrodinger_head.requires_grad_(False)


        
    def train_schrodinger_head(self, x):
        x = x.requires_grad_(True)
        _, x, psi = self.schrodinger_model(x)
        ##############################
        grad_psi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        grad_grad_psi = torch.autograd.grad(grad_psi, x, grad_outputs=torch.ones_like(grad_psi), create_graph=True, retain_graph=True)[0]

        # a = torch.max(x)
        # if a > self.schrodinger_model.a.data:
        #     self.schrodinger_model.a.data = a
        # else:
        #     a = self.schrodinger_model.a.data.item()

        mask_left = (x < self.schrodinger_model.a).float()
        mask_right = (x >= self.schrodinger_model.a).float()
        
        ###########
        residual_left = (-(self.h**2/(2*self.m))) * grad_grad_psi - self.E * psi
        residual_right = (-(self.h**2/(2*self.m))) * grad_grad_psi + self.V0 * psi - self.E * psi
        pde_loss = (mask_left * torch.abs(residual_left)**2 + mask_right * torch.abs(residual_right)**2).mean()
        ###########
        # if isinstance(a, torch.Tensor):
        #     a = a.item()
        # print(a)
        x_left = torch.full((x.shape[0],1), self.schrodinger_model.a-self.epsilon, requires_grad=True).to(x.device)
        x_right = torch.full((x.shape[0],1), self.schrodinger_model.a+self.epsilon, requires_grad=True).to(x.device)
        psi_left = self.schrodinger_model.schrodinger_head(x_left)
        psi_right = self.schrodinger_model.schrodinger_head(x_right)
        bc_value_loss = torch.abs(psi_left - psi_right).pow(2).mean()

        ###########
        grad_left = torch.autograd.grad(psi_left.sum(), x_left, create_graph=True)[0]
        grad_right = torch.autograd.grad(psi_right.sum(), x_right, create_graph=True)[0]
        bc_grad_loss = torch.abs(grad_left - grad_right).pow(2).mean()

        ###########
        x_boundary_left = torch.full((x.shape[0]//2,1), self.boundary_left).to(x.device)
        x_boundary_right = torch.full((x.shape[0]//2, 1), self.boundary_right).to(x.device)
        x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0).requires_grad_(True)
        psi_boundary = self.schrodinger_model.schrodinger_head(x_boundary)
        real_b, imag_b = psi_boundary[:, 0:1], psi_boundary[:, 1:2]
        bc_domain_loss = (real_b.pow(2).mean() + imag_b.pow(2).mean())

        print(f"pde_loss: {pde_loss.item()}, bc_value_loss: {bc_value_loss.item()}, bc_grad_loss: {bc_grad_loss.item()}, bc_domain_loss: {bc_domain_loss.item()}, total_loss: {pde_loss.item() + self.gamma* (bc_value_loss.item() + bc_grad_loss.item()) + self.delta*bc_domain_loss.item()}")

        if pde_loss.item() != 0:
            pde_loss = pde_loss/pde_loss.item()
        if bc_value_loss.item() != 0:
            bc_value_loss = bc_value_loss/bc_value_loss.item()
        if bc_grad_loss.item() != 0:
            bc_grad_loss = bc_grad_loss/bc_grad_loss.item()
        if bc_domain_loss.item() != 0:
            bc_domain_loss = bc_domain_loss/bc_domain_loss.item()

        total_schrodinger_loss = pde_loss + self.gamma* (bc_value_loss + bc_grad_loss) + self.delta*bc_domain_loss


        # print(f"pde_loss: {pde_loss.item()}, bc_value_loss: {bc_value_loss.item()}, bc_grad_loss: {bc_grad_loss.item()}, bc_domain_loss: {bc_domain_loss.item()}")


        self.optimizer_head.zero_grad()
        total_schrodinger_loss.backward()
        self.optimizer_head.step()

        return self.schrodinger_model.schrodinger_head, self.schrodinger_model.a, self.optimizer_head
    
    def visualize_schrodinger(self, device, epoch=0, save_path=None):
        x_test = torch.linspace(self.boundary_left, self.boundary_right, 1000).unsqueeze(1).to(device)
        with torch.no_grad():
            psi = self.schrodinger_model.schrodinger_head(x_test)
        psi = psi.cpu()
        x_test = x_test.cpu()
        plt.figure(figsize=(12, 6))
        plt.plot(x_test.numpy(), psi[:, 0].numpy(), label='Real Part')
        plt.plot(x_test.numpy(), psi[:, 1].numpy(), label='Imaginary Part')
        plt.axvline(0, color='r', linestyle='--', label='Potential Step (V0=1)')
        plt.xlabel('Position (x)')
        plt.ylabel('Wavefunction')
        plt.title(f'PINN Solution for E={self.E} (V0={self.V0})')
        # plt.title(f'PINN Solution for E={model.E} (V0={V0})')
        plt.legend()
        plt.grid(True)
        save_path = save_path + f'schrodinger_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def get_a(self, x, counter):
        _, x, psi = self.schrodinger_model(x)
        # print(x)
        if counter == 0:
            self.schrodinger_model.a = torch.mean(x).item()
        else:
            self.schrodinger_model.a = self.schrodinger_model.a*counter/(counter+1) + torch.mean(x).item()/(counter+1)
        
        if self.boundary_left > torch.min(x).item():
            self.boundary_left = torch.min(x).item()

        return self.schrodinger_model.a, self.boundary_left




