from backprop_modifier import *
from am import *

def normalize(array, min_value=0., max_value=1.):
    """Normalizes the numpy array to (min_value, max_value)
    Args:
        array: The numpy array
        min_value: The min value in normalized array (Default value = 0)
        max_value: The max value in normalized array (Default value = 1)
    Returns:
        The array normalized to range between (min_value, max_value)
    """
    arr_min = np.min(array)
    arr_max = np.max(array)
    normalized = (array - arr_min) / (arr_max - arr_min + K.epsilon())
    return (max_value - min_value) * normalized + min_value


def visualize_saliency_with_losses(input_tensor, losses, seed_input, wrt_tensor=None, grad_modifier=None, norm=False):
    """Generates an attention heatmap over the `seed_input` by using positive gradients of `input_tensor`
    with respect to weighted `losses`.
    This function is intended for advanced use cases where a custom loss is desired. For common use cases,
    refer to `visualize_class_saliency` or `visualize_regression_saliency`.
    For a full description of saliency, see the paper:
    [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps]
    (https://arxiv.org/pdf/1312.6034v2.pdf)
    Args:
        input_tensor: An input tensor of shape: `(samples, channels, image_dims...)` if `image_data_format=
            channels_first` or `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
        losses: List of ([Loss](vis.losses#Loss), weight) tuples.
        seed_input: The model input for which activation map needs to be visualized.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')
    Returns:
        The normalized gradients of `seed_input` with respect to weighted `losses`.
    """
    opt = Optimizer(input_tensor, losses, wrt_tensor=wrt_tensor, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)[1]

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(grads, axis=channel_idx)

    if norm:
        grads = normalize(grads)[0]
    else:
        grads = grads[0]
    return grads


def visualize_saliency(model, layer_idx, filter_indices, seed_input,
                       wrt_tensor=None, backprop_modifier=None, grad_modifier=None):
    """Generates an attention heatmap over the `seed_input` for maximizing `filter_indices`
    output in the given `layer_idx`.
    Args:
        model: The `keras.models.Model` instance. The model input shape must be: `(samples, channels, image_dims...)`
            if `image_data_format=channels_first` or `(samples, image_dims..., channels)` if
            `image_data_format=channels_last`.
        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        seed_input: The model input for which activation map needs to be visualized.
        wrt_tensor: Short for, with respect to. The gradients of losses are computed with respect to this tensor.
            When None, this is assumed to be the same as `input_tensor` (Default value: None)
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None)
        grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). By default `absolute`
            value of gradients are used. To visualize positive or negative gradients, use `relu` and `negate`
            respectively. (Default value = 'absolute')
    Example:
        If you wanted to visualize attention over 'bird' category, say output index 22 on the
        final `keras.layers.Dense` layer, then, `filter_indices = [22]`, `layer = dense_layer`.
        One could also set filter indices to more than one value. For example, `filter_indices = [22, 23]` should
        (hopefully) show attention map that corresponds to both 22, 23 output categories.
    Returns:
        The heatmap image indicating the `seed_input` regions whose change would most contribute towards
        maximizing the output of `filter_indices`.
    """
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    ]
    return visualize_saliency_with_losses(model.input, losses, seed_input, wrt_tensor, grad_modifier)
