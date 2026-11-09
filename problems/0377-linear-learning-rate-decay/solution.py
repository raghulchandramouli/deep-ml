def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    # Your code here
    

    if num_steps <= 1:
        return [initial_lr] * num_steps
    lr_list = []

    for step in range(num_steps):

        progress = step / (num_steps - 1)

        current_lr = initial_lr + progress * (end_lr - initial_lr)
        lr_list.append(current_lr)
    return lr_list