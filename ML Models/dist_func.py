import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse

def negative_bionomial_layer(x):
    '''
    Lamda function generating negative binomial parameters
    n and p from Dense(2) output
    Assume tensorflow 2 backend

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)
    
    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer
        
    Returns
    -------
    out_tensor : tf.Tensor
    '''

    #Get num of dimensions of input
    num_dims = len(x.get_shape())
    n, p = tf.unstack(x, num=2, axis=-1)

    #Add one dim to make right shape
    n = tf.expand_dims(n,-1)
    p = tf.expand_dims(p,-1)

    # n is positive, apply softplus
    n = tf.keras.activations.softplus(n)

    # 0 < p <1, apply sigmoid
    p = tf.keras.activations.sigmoid(p)

    out_tensor = tf.concat((n,p),axis=num_dims-1)

    return out_tensor

def negative_bionomial_loss(y_true,y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """  
    #Separate parameters
    n, p = tf.unstack(y_pred, num=2 ,axis = -1)

    #Add one dim to make it the right shape
    n = tf.expand_dims(n,-1)
    p = tf.expand_dims(p,-1)

    #Negative log likelihood
    nll = (
        tf.math.lgamma(n) 
        + tf.math.lgamma(y_true+1)
        - tf.math.lgamma(n+y_true)
        - n * tf.math.log(p)
        - y_true * tf.math.log(1-p)
    )

    return nll

def gaussian_distribution_layer(x):
    '''
    Lamda function generating gaussian distribution parameters
    n and p from Dense(2) output
    Assume tensorflow 2 backend

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)
    
    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer
        
    Returns
    -------
    out_tensor : tf.Tensor
    '''

    import tensorflow as tf
     
    #x1 = tf.reshape(x,[-1,2,96])
    mu,sigma = tf.split(x, num_or_size_splits=2, axis=1)
     
    # Sigma always positive, apply softplus
    sigma = tf.keras.activations.softplus(sigma)

    out_tensor = tf.concat((mu,sigma),axis=1)

    return out_tensor

def gaussian_loss(y_true,y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """  

    import tensorflow as tf

    #Separate parameters
    
    mu,sigma = tf.split(y_pred, num_or_size_splits=2, axis=1)
     
    #Negative log likelihood
    #https://stats.stackexchange.com/questions/521091/optimizing-gaussian-negative-log-likelihood
    nll = tf.math.reduce_sum(
        (1/2)
        *(tf.math.log(tf.math.maximum(sigma**2,1e-12))
        +((y_true-mu)**2)/tf.math.maximum(sigma**2,1e-12)), axis=-1
    )
    
    nll = tf.math.reduce_mean(nll,axis=-1)
    
    #nll = tf.clip_by_value(nll,-1e8,1e8)

    if(tf.math.is_nan(nll) | (nll>1e8)):
        nll = nll*0+1e8

    return nll


def RMSE_gaussian_metric(y_true,y_pred):

    mu,_ = tf.split(y_pred, num_or_size_splits=2, axis=1)

    return tf.math.sqrt(mse(mu,y_true))

