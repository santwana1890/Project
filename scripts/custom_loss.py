import tensorflow as tf

#Custom Loss function to help with class weights

def weighted_sparse_categorical_cross_entropy(class_weight):
  def loss(y_obs, y_pred):
    y_obs = tf.dtypes.cast(y_obs,tf.int32)
    hothot = tf.one_hot(tf.reshape(y_obs, [-1]), depth=len(class_weight))
    print(hothot)
    weight = tf.math.multiply(class_weight, hothot)
    weight = tf.reduce_sum(weight, axis =1)
    losses = tf.compat.v1.losses.sparse_softmax_cross_entropy( labels = y_obs, logits=y_pred,weights= weight )
    return losses
  return loss
