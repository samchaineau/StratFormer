import tensorflow as tf

def single_accuracy(y_true: dict, y_pred: list) -> dict:
    """Compute Team, Position and Mask metrics for a batch of single sequence:
        - binary accuracy for Team prediction
        - categorical accuracy for Position classification
        - topk accuracy 3 for Position classification
        - categorical accuracy for Mask prediction
        - topk accuracy 3 for Mask prediction

    Args:
        y_true (dict): a dict with keys: Team, Pos and Mask with the true label for each values
        y_pred (list): a list with elements: Team, Pos and Mask with the predicted label for each element

    Returns:
        dict: dict with single sequence metrics
    """
    pred_team = tf.cast(tf.round(y_pred[0]), dtype ="int64")
    team_accuracy = tf.keras.metrics.binary_accuracy(y_true["Team"], pred_team)
    team_accuracy = tf.reduce_mean(team_accuracy)

    pos_accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true["Pos"], y_pred[1])
    pos_accuracy = tf.reduce_mean(pos_accuracy)

    pos_topK = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true["Pos"], y_pred[1], k = 3)
    pos_topK = tf.reduce_mean(pos_topK)
    
    mask_accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true["Mask"], y_pred[2])
    mask_accuracy = tf.reduce_mean(mask_accuracy)
    
    mask_topK = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true["Mask"], y_pred[2], k = 3)
    mask_topK = tf.reduce_mean(mask_topK)

    return {"Team_Accuracy" : team_accuracy,
            "Pos_Accuracy" : pos_accuracy,
            "Pos_topK" : pos_topK, 
            "Mask_Accuracy" : mask_accuracy, 
            "Mask_topK" : mask_topK}

def compute_accuracy(y_true: dict, y_pred: list) -> dict:
    """Compute metrics for a single batch. A batch is made of pairs of sequences.

    Each sequence is used to compute the single metric. Then the SamePlay metric is computed using both sequence.
    Every metrics computed by sequence are then averaged into a single value per metrics.

    Args:
        y_true (dict): a dict with keys: Team, Pos and Mask with the true label for each values
        y_pred (list): a list with elements: Team, Pos and Mask with the predicted label for each element

    Returns:
        dict: dict with every metrics
    """
    acc1 = single_accuracy(y_true["Y1"], y_pred[1][1:])
    acc2 = single_accuracy(y_true["Y2"], y_pred[2][1:])

    sameplay_accuracy = tf.keras.metrics.binary_accuracy(y_true["Strat"], y_pred[0])

    team_accuracy = (acc1["Team_Accuracy"]+acc2["Team_Accuracy"])/2
    
    pos_accuracy = (acc1["Pos_Accuracy"]+acc2["Pos_Accuracy"])/2
    
    pos_topK = (acc1["Pos_topK"]+acc2["Pos_topK"])/2
    
    mask_accuracy = (acc1["Mask_Accuracy"]+acc2["Mask_Accuracy"])/2
    
    mask_topK = (acc1["Mask_topK"]+acc2["Mask_topK"])/2

    return {"SamePlay_Accuracy" : sameplay_accuracy,
            "Team_Accuracy" : team_accuracy,
            "Pos_Accuracy" : pos_accuracy,
            "Pos_topK" : pos_topK, 
            "Mask_Accuracy" : mask_accuracy, 
            "Mask_topK" : mask_topK}