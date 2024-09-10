from art.defences.preprocessor import GaussianAugmentation
import torch

def Counter_Samples(iters=10,model x_preprocessed, k=0.03, sigma=0.01):
    """
    The implemetation of CounterSamples - this is only one implementation, and it could be implemented in multiple ways. 
    It acts as a preprocessor, taking preprocessed samples (x_preprocessed), the number of optimization iterations (iters), 
    and the step size (k), and returns the optimized (corrected) samples.

    Parameters:

    k (float): Step size for each optimization iteration, guiding the reduction of the loss.
    sigma (float): Magnitude of the Gaussian noise added to the samples.
    model (torch model): The target model to be optimized.
    iters (int): Number of optimization iterations to perform.
    """

    # Apply noise first
    use_cuda = True
    gaus = GaussianAugmentation(augmentation=False, sigma=0.01)
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    x_preprocessed = torch.from_numpy(gaus(x_preprocessed.detach().cpu().numpy())[0]).to(device)
    x_preprocessed.requires_grad_(True)
    x_preprocessed.retain_grad()
    loss = nn.CrossEntropyLoss(reduction='none')
    for iter in range(iters):
        # predicting labels
        model_output = model(x_preprocessed)
        true_labels_indexes = torch.argmax(model_output, dim=1)
        loss_comp = loss(model_output, true_labels_indexes)
        loss_comp.backward(torch.ones_like(loss_comp))
        # update the samples.
        x_preprocessed = x_preprocessed - k * x_preprocessed.grad # No normalization.
        x_preprocessed.retain_grad()
    return x_preprocessed