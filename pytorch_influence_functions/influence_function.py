'''
Codes from github: https://github.com/nimarb/pytorch_influence_functions
Reference: github repo release-influence
'''

import torch
from torch.autograd import grad
from pytorch_influence_functions.utils import display_progress
from tqdm import tqdm, trange
from random import choice, randint
import torch.nn.functional as F

def I_pert_loss(s_test, model, x, t, device=torch.device('cpu'), all_param=True):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        s_test: torch tensor, pre-compute hvp, grad_theta * H^-1
        model: torch NN, model used to evaluate the dataset
        x, t: perturbed sample
        device: torch.device
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""

    x, t = x.to(device), t.to(device)
    x.requires_grad = True
    
    model.eval()
    y = model(x)
    loss = calc_loss(y, t)
    
    if all_param:
        for param in model.parameters():
            param.requires_grad = True
        
    
    params = [ p for p in model.parameters() if p.requires_grad ]
    
    # first backprop by theta
    first_grads = grad(loss, params, retain_graph=True, create_graph=True)
        
    # do hessian  vector product
    
    elemwise_products = [
        torch.mul(grad_elem, v_elem.detach())
        for grad_elem, v_elem in zip(first_grads, s_test) if grad_elem is not None
    ]
        
    # Second backprop by x
    #grads_with_none = grad(sum([elemwise_product.sum() for elemwise_product in elemwise_products]), x)
    grads_with_none = grad(elemwise_products, x, grad_outputs=elemwise_products)
    
    I_pert = [
        grad_elem if grad_elem is not None \
        else torch.zeros_like(x) \
        for x, grad_elem in zip(params, grads_with_none)
    ]
    
    return I_pert

def avg_s_test(test_loader, model, z_loader, device=torch.device('cpu'), damp=0.0, scale=10.0,
           recursion_depth=5000, all_param=True, original_label=False, sample_num=1):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        test_loader: dataloader, contains all test data for averaging
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        device: torch.device
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
        
    parameter_gradients = None
    num_datapoints = len(test_loader.dataset)
    
    if all_param:
        for param in model.parameters():
            param.requires_grad = True
    
    for inputs, targets, *others in tqdm(test_loader, desc='Calc. the averaging gradients'):
        
        if original_label:
            targets = others[-2]
        
        model.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = calc_loss(outputs, targets)
        loss.backward()
        
        if all_param:
            params = model.parameters()
        else:
            params = [param for param in model.parameters() if param.requires_grad]
        if parameter_gradients is None:
            parameter_gradients = [p.grad / num_datapoints for p in params]
        else:
            parameter_gradients = [(p.grad / num_datapoints + g) for p, g 
                                   in zip(params, parameter_gradients)]
    
    # set C(θ) as averaging_gradients
    v = parameter_gradients
    h_estimate = v.copy()
    
    z_iter = iter(z_loader)

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in tqdm(range(recursion_depth), desc='Calc. s_test recursions'):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        try:
            x, t, *others = next(z_iter) # random choise
        except StopIteration:
            z_iter = iter(z_loader)
            x, t, *others = next(z_iter)
        x, t = x.to(device), t.to(device)
        model.eval()
        y = model(x)
        loss = calc_loss(y, t)
        params = [ p for p in model.parameters() if p.requires_grad ]
        hv = hvp(loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate = [
            (_v + (1 - damp) * _h_e - _hv / scale).detach()
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        #display_progress("Calc. s_test recursions: ", i, recursion_depth)
    
    inverse_hvp = [b/sample_num for b in h_estimate]
    #inverse_hvp = h_estimate
    
    return inverse_hvp

def s_test(z_test, t_test, model, z_loader, device=torch.device('cpu'), damp=0.0, scale=10.0,
           recursion_depth=5000, sample_num=1):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        device: torch.device
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, device)
    h_estimate = v.copy()
    
    z_iter = iter(z_loader)

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in tqdm(range(recursion_depth), desc='Calc. s_test recursions'):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        try:
            x, t, *others = next(z_iter) # random choise
        except StopIteration:
            z_iter = iter(z_loader)
            x, t, *others = next(z_iter)
        x, t = x.to(device), t.to(device)
        model.eval()
        y = model(x)
        loss = calc_loss(y, t)
        params = [ p for p in model.parameters() if p.requires_grad ]
        hv = hvp(loss, params, h_estimate)
        # Recursively caclulate h_estimate
        h_estimate = [
            (_v + (1 - damp) * _h_e - _hv / scale).detach()
            for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        #display_progress("Calc. s_test recursions: ", i, recursion_depth)
    
    inverse_hvp = [a/sample_num for a in h_estimate]
    #inverse_hvp = h_estimate
    
    return inverse_hvp

def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    loss = torch.nn.CrossEntropyLoss()(y, t)
    return loss

def grad_z(z, t, model, device):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        device: torch.device

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    model.zero_grad()
    # initialize
    z, t = z.to(device), t.to(device)
    y = F.normalize(F.softmax(model(z), dim=1))
    loss = calc_loss(y, t) # We use CrossEntropyLoss
    
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    loss.backward()
    return [p.grad for p in params]

def hvp(ys, xs, v):
    # Reference：github repo 'influence-release'
    
    """Multiply the Hessians of ys and xs by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * x^T A x then hvp(ys, xs, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        ys: scalar/tensor, for example the output of the loss function
        xs: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `xs` have a different length."""
    if len(xs) != len(v):
        raise(ValueError("xs and v must have the same length."))

    # First backprop
    first_grads = grad(ys, xs, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = [
        torch.mul(grad_elem, v_elem.detach())
        for grad_elem, v_elem in zip(first_grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = grad(elemwise_products, xs, 
                           grad_outputs=[torch.zeros_like(elemwise_product) 
                                         for elemwise_product in elemwise_products])
    return_grads = [
        grad_elem if grad_elem is not None \
        else torch.zeros_like(x) \
        for x, grad_elem in zip(xs, grads_with_none)
    ]

    return return_grads

