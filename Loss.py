import numpy as np
import torch
from torch import nn

""" WGAN """
def discriminator_loss(y_pred, y_pred_fake):
    # During discriminator forward-backward-update
    D_loss = -(torch.mean(y_pred) - torch.mean(y_pred_fake))
    return D_loss

def generator_loss(y_pred, y_pred_fake):
    # During generator forward-backward-update
    G_loss = -torch.mean(y_pred_fake)
    return G_loss
