# TensorFlow 2.X

The code in this repository was originally done using the version 1.13 of TensorFlow, however it can be easily adjusted to work with TensorFlow version 2.X. In order to perform this modification some alterations are need:

1. Change any `import keras` for `import tensorflow.keras`
2. The following code are no longer needed:
```python
try:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    K.set_session(sess)
except:
    pass
```
This code can be found in the `train.py` and `inference.py` scripts.

