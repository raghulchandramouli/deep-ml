def calculate_parameters(layers: list[dict]) -> int:
    total_params = 0

    for layer in layers:
        layer_type = layer.get("type", "")

        # Dense / Fully Connected Layer
        if layer_type == "dense":
            input_size = layer["input_size"]
            output_size = layer["output_size"]
            bias = layer.get("bias", True)

            total_params += (input_size * output_size)

            if bias:
                total_params += output_size

        # Conv2D Layer
        elif layer_type == "conv2d":
            in_channels = layer["in_channels"]
            out_channels = layer["out_channels"]
            kernel_size = layer["kernel_size"]

            bias = layer.get("bias", True)

            # kernel params
            total_params += (
                in_channels
                * out_channels
                * kernel_size
                * kernel_size
            )

            if bias:
                total_params += out_channels

    return total_params