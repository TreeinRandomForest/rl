import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation):
        super(PolicyNet, self).__init__()
        
        self.N_inputs = N_inputs
        self.N_outputs = N_outputs
        
        self.N_hidden_layers = N_hidden_layers
        self.N_hidden_nodes = N_hidden_nodes
        
        self.layer_list = nn.ModuleList([]) #use just as a python list
        for n in range(N_hidden_layers):
            if n==0:
                self.layer_list.append(nn.Linear(N_inputs, N_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(N_hidden_nodes, N_hidden_nodes))
        
        self.output_layer = nn.Linear(N_hidden_nodes, N_outputs)
        
        self.activation = activation
        self.output_activation = output_activation
        
    def forward(self, inp):
        out = inp.reshape(inp.shape[0], -1)
        for layer in self.layer_list:
            out = layer(out)
            out = self.activation(out)
            
        out = self.output_layer(out)
        if self.output_activation is not None:
            pred = self.output_activation(out)
        else:
            pred = out
        
        return pred