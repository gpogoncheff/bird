import torch.nn as nn

class TeacherStudent(nn.Module):
    """
    Teacher-Student module for distillation process

    Params:
        teacher_model (nn.Module): teacher model
        student_model (nn.Module): student model
        teacher_alignment_layer (str): name of guiding teacher layer
        student_alignment_layer (str): name of guided student layer
        teacher_projection (nn.Module): model projecting teacher representations
        student_projection (nn.Module): model projecting student representations
        freeze_teacher (bool): Freeze teacher at initialization
        freeze_teacher_bn (bool): Freeze teacher's running statistics
        teacher_eval (bool): Set teacher to eval mode
    """
    def __init__(
        self, 
        teacher_model, 
        student_model, 
        teacher_alignment_layer, 
        student_alignment_layer, 
        teacher_projection=nn.Identity(),
        student_projection=nn.Identity(),
        freeze_teacher=True,
        freeze_teacher_bn=False,
        teacher_eval=False,
    ):
        super(TeacherStudent, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_alignment_layer = teacher_alignment_layer
        self.student_alignment_layer = student_alignment_layer
        self.teacher_projection = teacher_projection
        self.student_projection = student_projection
        if freeze_teacher:
            self.freeze_teacher_model(freeze_teacher_bn)
        if teacher_eval:
            self.teacher_model.eval()

        # register hooks to retain teacher and student 
        # representations at a given layer
        self.teacher_activations = {}
        self.student_activations = {}

        hook_layer = None
        for name, module in self.teacher_model.named_modules():
            if name == teacher_alignment_layer:
                hook_layer = module
                break
        assert hook_layer is not None, f"Layer {teacher_alignment_layer} not found in teacher model"
        hook_layer.register_forward_hook(self.get_activations("features", is_teacher=True))

        hook_layer = None
        for name, module in self.student_model.named_modules():
            if name == student_alignment_layer:
                hook_layer = module
                break
        assert hook_layer is not None, f"Layer {student_alignment_layer} not found in student model"
        hook_layer.register_forward_hook(self.get_activations("features", is_teacher=False))

    def get_activations(self, activation_name, is_teacher=False):
        def hook(model, input, output):
            if is_teacher:
                self.teacher_activations[activation_name] = output
            else:
                self.student_activations[activation_name] = output
        return hook

    def set_is_training(self):
        self.student_model.train()
        self.teacher_projection.train()
        self.student_projection.train()

    def set_is_eval(self):
        self.student_model.eval()
        self.teacher_projection.eval()
        self.student_projection.eval()

    def freeze_teacher_model(self, freeze_bn=True):
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        if freeze_bn:
            for module in self.teacher_model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    module.track_running_stats = False
    
    def forward(self, x):
        teacher_output = self.teacher_model(x)
        student_output = self.student_model(x)
        teacher_features = self.teacher_activations["features"]
        student_features = self.student_activations["features"]
        teacher_feature_projection = self.teacher_projection(teacher_features)
        student_feature_projection = self.student_projection(student_features)
        return (teacher_output, student_output), (teacher_feature_projection, student_feature_projection)