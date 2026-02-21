from bird.models.alexnet import AlexNet32
from bird.models.densenet import (
    DenseNet121,
    DenseNet169,
    DenseNet201
)
from bird.models.mobilenet import MobileNetV2
from bird.models.resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152
)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

def get_fixed_model_state_dict(state_dict):
    """
    Resolves parameter names in state dict for models that
    were compiled.

    Args:
        state_dict: model state dict

    Returns:
        state_dict: state_dict with resolved names
    """
    in_state_dict = state_dict
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if not all(src_key == dest_key for src_key, dest_key in pairings):
        out_state_dict = {}
        for src_key, dest_key in pairings:
            out_state_dict[dest_key] = in_state_dict[src_key]
        state_dict = out_state_dict
    return state_dict

def get_model(model_name, num_classes, state_dict=None):
    model_name = model_name.lower()
    if model_name == "alexnet":
        model = AlexNet32(num_classes=num_classes)
    elif model_name == "densenet121":
        model = DenseNet121(num_classes=num_classes)
    elif model_name == "densenet169":
        model = DenseNet169(num_classes=num_classes)
    elif model_name == "densenet201":
        model = DenseNet201(num_classes=num_classes)
    elif model_name == "mobilenetv2":
        model = MobileNetV2(num_classes=num_classes)
    elif model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = ResNet34(num_classes=num_classes)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif model_name == "resnet101":
        model = ResNet101(num_classes=num_classes)
    elif model_name == "resnet152":
        model = ResNet152(num_classes=num_classes)
    else:
        raise NotImplementedError(f"Model {model_name} not yet implemented")
    
    if state_dict is not None:
        state_dict = get_fixed_model_state_dict(state_dict)
        model.load_state_dict(state_dict)
        
    return model