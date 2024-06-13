import torch

'''
For better understanding of backdoor features.

Reference: https://zhuanlan.zhihu.com/p/375811657, https://zhuanlan.zhihu.com/p/362985275
'''

'''
定义好模型后,假设我们提取 avgpool前的feature,即conv1后的feature:
'''

def get_feas_by_hook_foward(model: torch.nn.Module, layer_name: str):
    """
    提取Conv2d后的feature,我们需要遍历模型的module,然后找到Conv2d,把hook函数注册到这个module上;
    这就相当于告诉模型,我要在Conv2d这一层,用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d,所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    features_in_hook = []
    features_out_hook = []
    handles = []
    
    # -------------------- 第一步：定义接收feature的函数 ---------------------- #
    # 这里定义了一个函数hook，用来有接收feature。
    def hook(module: torch.nn.Module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None
    
    for name, module in model.named_modules():
        if name == layer_name:
            # ---------- 第二步：注册hook,告诉模型我将在哪些层提取feature -------- #
            handle = module.register_forward_hook(hook=hook)
            handles.append(handle)

    return features_in_hook, features_out_hook, handles

def handles_remove(handles):
    for handle in handles:
        handle.remove()

def get_classifier_from_model(model: torch.nn.Module):
    return list(model.named_children())[-1][1]