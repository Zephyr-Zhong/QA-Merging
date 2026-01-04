import torch
import torch.nn.functional as F

def contrastive_loss_token(features_0, features_pos, features_neg, temperature=0.05):

    features_0, features_pos, features_neg = F.normalize(features_0, dim=1), F.normalize(features_pos, dim=1), F.normalize(features_neg, dim=1)

    # cosine similarity = dot product of two normalized vectors
    pos = torch.exp(torch.sum(features_0*features_pos, dim=-1).to(torch.float32) / temperature)
    neg = torch.exp(torch.sum(features_0*features_neg, dim=-1).to(torch.float32) / temperature)

    Ng = neg.sum(dim=-1)
    loss = (-torch.log(pos / (Ng+pos))).mean()

    return loss
