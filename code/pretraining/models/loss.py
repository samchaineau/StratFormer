import tensorflow as tf
from focal_loss import sparse_categorical_focal_loss

def single_loss(y_true: dict, y_pred: list) -> dict:
    """Compute Team, Position and Mask loss for a batch of single sequence:
        - binary crossentropy for Team prediction
        - categorical focal crossentropy for Position classification
        - categorical focal crossentropy for Mask prediction

    Args:
        y_true (dict): a dict with keys: Team, Pos and Mask with the true label for each values
        y_pred (list): a list with elements: Team, Pos and Mask with the predicted label for each element

    Returns:
        dict: a dict with the losses computed for a single batch
    """
    teamloss = tf.keras.losses.binary_crossentropy(y_true= y_true["Team"], y_pred= y_pred[0], label_smoothing= 0.2)
    teamloss = tf.reduce_mean(teamloss)

    poscce = sparse_categorical_focal_loss(y_true= y_true["Pos"], y_pred= y_pred[1], gamma = 2)
    poscce = tf.reduce_mean(poscce)
    posloss = poscce

    maskcce = sparse_categorical_focal_loss(y_true= y_true["Mask"], y_pred= y_pred[2], gamma = 2)
    maskcce = tf.reduce_mean(maskcce)
    maskloss = maskcce

    return {"Team_Loss": teamloss, "Pos_Loss": posloss, "Mask_Loss": maskloss}


def loss(y_true: dict, y_pred: list) ->dict:
    """Compute loss for a single batch. A batch is made of pairs of sequences.

    Each sequence is used to compute the single loss. Then the SamePlay loss is computed using both sequence.
    Total loss is the weighted sum of each sub loss.

    Args:
        y_true (dict): a dict with keys: Team, Pos and Mask with the true label for each values
        y_pred (list): a list with elements: Team, Pos and Mask with the predicted label for each element

    Returns:
        dict: dict with Total loss and each sub loss
    """
    loss_1 = single_loss(y_true["Y1"], y_pred[1][1:])
    loss_2 = single_loss(y_true["Y2"], y_pred[2][1:])
    
    sameplayloss = tf.keras.losses.binary_crossentropy(y_true["Strat"], y_pred[0], label_smoothing= 0.2)
    sameplayloss = tf.reduce_mean(sameplayloss)

    teamloss = (loss_1["Team_Loss"]+loss_2["Team_Loss"])/2
    posloss = (loss_1["Pos_Loss"]+loss_2["Pos_Loss"])/2
    maskloss = (loss_1["Mask_Loss"]+loss_2["Mask_Loss"])/2
    total_loss = 3*sameplayloss + (1.2*teamloss) + (0.8*posloss) + (0.4*maskloss)

    return {"Total_Loss": total_loss, 
            "SamePlay_Loss": sameplayloss, 
            "Team_Loss": teamloss, 
            "Pos_Loss": posloss, 
            "Mask_Loss": maskloss}