import tensorflow as tf

from .jacobi_polynomials import jacobi_polynomial


class FractionalJacobiNeuralBlock(tf.keras.layers.Layer):
    """
    Fractional Jacobi Neural Block layer for TensorFlow.

    This layer computes a custom transformation using the Jacobi polynomial.

    Attributes:
        degree (int): Degree of the Jacobi polynomial.
    """

    def __init__(self, degree, **kwargs):
        """
        Initialize the Fractional Jacobi Neural Block.

        Args:
            degree (int): Degree of the Jacobi polynomial.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super(FractionalJacobiNeuralBlock, self).__init__(**kwargs)
        self.degree = degree
        self.jacobi_polynomial = tf.function(jacobi_polynomial)

    def build(self, input_shape):
        """
        Create the weights of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.alpha = self.add_weight(
            name="alpha",
            initializer="ones",
            trainable=True,
            shape=(1,)
        )
        self.beta = self.add_weight(
            name="beta",
            initializer="ones",
            trainable=True,
            shape=(1,)
        )
        self.zeta = self.add_weight(
            name="gamma",
            initializer="zeros",
            trainable=True,
            shape=(1,)
        )
        super(FractionalJacobiNeuralBlock, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Jacobi polynomial transformation.
        """
        normalized_alpha = tf.keras.activations.elu(self.alpha, 1)
        normalized_beta = tf.keras.activations.elu(self.beta, 1)
        normalized_zeta = tf.keras.activations.sigmoid(self.zeta)
        inputs = tf.keras.activations.sigmoid(inputs)

        return self.jacobi_polynomial(
            inputs, self.degree, normalized_alpha, normalized_beta, normalized_zeta, 0, 1
        )
