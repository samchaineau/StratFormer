import tensorflow as tf

class TemporalModel(tf.keras.Model):
    """Model converting timesteps into dense embeddings.
    """
    def __init__(self):
        super(TemporalModel, self).__init__()

        self.Embedding = tf.keras.layers.Embedding(input_dim = 25, output_dim = 256, mask_zero = True, name = "temporal_embedding")

    def call(self, x):
        embed = self.Embedding(x)
        return embed

class TokenModel(tf.keras.Model):
    """Model converting zones and tokens into embeddings.
    """
    def __init__(self):
        super(TokenModel, self).__init__()
        self.Embedding = tf.keras.layers.Embedding(input_dim = 6067, output_dim = 256, mask_zero = True, name = "token_embedding")

    def call(self, x):
        embed = self.Embedding(x)
        return embed

class PreProcessInput(tf.keras.Model):
    """Model converting tokens, zones and timesteps into embeddings then add them respectively. 
    Returns final embeddings.
    """
    def __init__(self):
        super(PreProcessInput, self).__init__()

        self.Temporal = TemporalModel()
        self.Zone = TokenModel()

        self.Add = tf.keras.layers.Add()

    def call(self, x):
        Temp_Embed = self.Temporal(x["temporal_id"])
        Zone_Embed = self.Zone(x["zone_id"])

        added = self.Add([Temp_Embed, Zone_Embed]) 
        return added
    
class AttentionBlock(tf.keras.Model):
    """Single attention block. Compute the attention scores of the input, apply matmul of the scores to the value and add the the output to the input.
    """
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.Attention = tf.keras.layers.AdditiveAttention(name = "attention", dropout = 0.15)
        self.Add = tf.keras.layers.Add()
        self.Norm = tf.keras.layers.BatchNormalization()

    def call(self, x, return_attention = False):
        attention, scores = self.Attention([x, x], return_attention_scores = True)
        if return_attention == True:
            return scores
        else:
            attention = self.Drop(attention)
            attention = self.Norm(attention)
            output = self.Add([x, attention])
            return output

class StratHead(tf.keras.Model):
    """Classification head to predict whether two trajectories are drawn from the same play.
    Inspired by the paper Sentence-BERT where both embeddings are concatenated with their absolute differences. 
    """
    def __init__(self):
        super(StratHead, self).__init__()
        self.Conc = tf.keras.layers.Concatenate()
        self.Diff = tf.keras.layers.Subtract()
        self.Classification = tf.keras.layers.Dense(1, activation = "sigmoid", use_bias = True)

    def call(self, x):
        spe1 = x[0][:,0,:]
        spe2 = x[1][:,0,:]
        diff = self.Diff([spe1, spe2])
        abs_diff = tf.abs(diff)
        concatenated = self.Conc([spe1, spe2, abs_diff])
        pred = self.Classification(concatenated)
        return pred

class TeamPredictionHead(tf.keras.Model):
    """Classification head to predict whether the trajectory is from an offensive or defensive player.
    """
    def __init__(self):
        super(TeamPredictionHead, self).__init__()
        self.Dense1_team = tf.keras.layers.Dense(128, activation = "relu", name = "embedding_team")
        self.Dense2_team = tf.keras.layers.Dense(1, activation = "sigmoid", name = "predicting_team")

    def call(self, x):
        densed1_team = self.Dense1_team(x)
        predicted_team = self.Dense2_team(densed1_team)
        return predicted_team

class PosPredictionHead(tf.keras.Model):
    """Classification head to predict the position of the player.
    """
    def __init__(self):
        super(PosPredictionHead, self).__init__()
        self.Dense1_pos = tf.keras.layers.Dense(128, activation = "relu", name = "embedding_pos")
        self.Dense2_pos = tf.keras.layers.Dense(16, activation = "softmax", name = "predicting_pos")

    def call(self, x):
        densed1_pos = self.Dense1_pos(x)
        predicted_pos = self.Dense2_pos(densed1_pos)
        return predicted_pos

class TrajCompletionHead(tf.keras.Model):
    """Classification head to predict the masked zones in the trajectory.
    """
    def __init__(self):
        super(TrajCompletionHead, self).__init__()
        self.Dense1_mask = tf.keras.layers.Dense(128, activation = "relu", name = "embedding_mask")
        self.Dense2_mask = tf.keras.layers.Dense(6062, activation = "softmax", name = "predicting_mask")

    def call(self, x):
        masks = tf.expand_dims(x[1]["mask_id"], axis =2)
        masks = tf.cast(masks, dtype = "float32")
        masked = tf.multiply(masks, x[0])
        mask_token = tf.reduce_sum(masked, axis = 1)
        densed1_mask = self.Dense1_mask(mask_token)
        predicted_mask = self.Dense2_mask(densed1_mask)
        return predicted_mask