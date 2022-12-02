import torch
import numpy as np

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def pesudo_loss_fun(y_true, y_pred, pseudos):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    pseudos: [1, 1, 1, 0, 0, 0] 1是伪标签， 0是非伪标签
    """

    pseudos = torch.tensor(np.array(pseudos))
    pseudo = list(np.where(pseudos == 1)[0])
    normal = list(np.where(pseudos == 0)[0])
    loss = 0.0
    if len(pseudo) != 0:
        y_true1, y_pred1 = y_true[pseudo], y_pred[pseudo]
        batch_size, ent_type_size = y_true1.shape[:2]
        y_true1 = y_true1.reshape(batch_size * ent_type_size, -1)
        y_pred1 = y_pred1.reshape(batch_size * ent_type_size, -1)
        loss1 = multilabel_categorical_crossentropy(y_true1, y_pred1)
        loss += loss1 * 0.5

    if len(normal) != 0:
        y_true2, y_pred2 = y_true[normal], y_pred[normal]
        batch_size, ent_type_size = y_true2.shape[:2]
        y_true2 = y_true2.reshape(batch_size * ent_type_size, -1)
        y_pred2 = y_pred2.reshape(batch_size * ent_type_size, -1)
        loss2 = multilabel_categorical_crossentropy(y_true2, y_pred2)
        loss += loss2

    return loss


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss
