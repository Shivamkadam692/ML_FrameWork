import matplotlib.pyplot as plt

def plot_loss_curve(loss_history, title="Training Loss Curve", xlabel="Iterations", ylabel="Loss"):
    """Plot the training loss curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, color='blue', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
