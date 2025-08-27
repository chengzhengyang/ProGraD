import torch
import torch.nn as nn

from .criterion import SetCriterion
from .group_matcher import build_group_matcher
from .models import GADTR


def build_model(args):
    model = GADTR(args)

    # Get backbone reference
    backbone = model.backbone[0]

    # Check if we should fine-tune the backbone
    finetune_backbone = args.backbone == 'finetune'
    if finetune_backbone:
        # Fine-tuning mode: separate learning rates for backbone and decoder
        print("Mode: Fine-tuning backbone")

        # Collect parameters - only backbone and decoder
        backbone_params = []
        decoder_params = []

        # Collect backbone parameters
        for param in backbone.model.parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # Collect decoder parameters
        backbone_param_ids = {id(p) for p in backbone_params}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Skip backbone parameters (already collected)
            if id(param) in backbone_param_ids:
                continue
            # All remaining trainable parameters are decoder parameters
            decoder_params.append(param)

        # Print parameter counts
        print(f"\nParameter Summary:")
        print(f"Backbone parameters: {sum(p.numel() for p in backbone_params):,}")
        print(f"Decoder parameters: {sum(p.numel() for p in decoder_params):,}")
        print(f"Total trainable: {sum(p.numel() for p in backbone_params + decoder_params):,}")

        # Create parameter groups - only backbone and decoder
        # Use backbone_lr_ratio to control the ratio (default: 0.1)
        backbone_lr_ratio = getattr(args, 'backbone_lr_ratio', 0.1)

        param_groups = [
            {
                'params': backbone_params,
                'lr': getattr(args, 'backbone_lr', args.lr * backbone_lr_ratio),
                'weight_decay': args.weight_decay,
                'name': 'backbone'
            },
            {
                'params': decoder_params,
                'lr': getattr(args, 'decoder_lr', args.lr),  # Default: same as base LR
                'weight_decay': args.weight_decay,
                'name': 'decoder'
            }
        ]

        # Print learning rates
        print(f"Backbone LR ratio: {backbone_lr_ratio}")
        print(f"Backbone LR: {param_groups[0]['lr']:.2e}")
        print(f"Decoder LR: {param_groups[1]['lr']:.2e}")

        # Set optimizer parameters for parameter groups
        optimizer_params = param_groups

        # Scheduler - handle different base LRs for different groups
        base_lrs = [group['lr'] for group in param_groups]
        max_lrs = [base_lr * (args.max_lr / args.lr) for base_lr in base_lrs]
        scheduler_params = {
            'base_lr': base_lrs,
            'max_lr': max_lrs
        }

        # For parameter groups, don't pass lr/weight_decay as they're defined in each group
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optimizer_params, betas=(0.9, 0.999), eps=1e-8)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optimizer_params, momentum=0.9, nesterov=True)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_params, betas=(0.9, 0.999), eps=1e-8)

    else:
        # Frozen backbone mode: freeze backbone and only train decoder
        print("Mode: Frozen backbone - only training decoder")

        # Freeze backbone parameters
        for param in backbone.model.parameters():
            param.requires_grad = False

        # Analyze all parameters in the backbone
        trainable_params = []
        non_trainable_params = []

        for name, param in backbone.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.shape, param.numel()))
            else:
                non_trainable_params.append((name, param.shape, param.numel()))

        print("Trainable backbone parameters:")
        for name, shape, numel in trainable_params:
            print(f"{name}: {shape}, {numel:,}")

        trainable_model_params = [param for param in model.parameters() if param.requires_grad]
        print(f"\nTotal trainable parameters: {sum(p.numel() for p in trainable_model_params):,}")

        # Set optimizer parameters for single learning rate
        optimizer_params = trainable_model_params

        # Simple scheduler parameters for single learning rate
        scheduler_params = {
            'base_lr': args.lr,
            'max_lr': args.max_lr
        }

        # For single parameter list, pass lr and weight_decay explicitly
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optimizer_params, lr=args.lr, momentum=0.9, nesterov=True,
                                        weight_decay=1e-4)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=args.weight_decay)

    # Create scheduler (shared code)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=scheduler_params['base_lr'],
        max_lr=scheduler_params['max_lr'],
        step_size_up=args.lr_step,
        step_size_down=args.lr_step_down,
        mode='triangular2',
        cycle_momentum=False
    )

    # Loss setup code (shared)
    losses = ['labels', 'cardinality']
    group_losses = ['group_labels', 'group_cardinality', 'group_code', 'group_consistency']

    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef

    group_matcher = build_group_matcher(args)
    criterion = SetCriterion(args.num_class, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, group_losses=group_losses, group_matcher=group_matcher, args=args)
    criterion.cuda()

    return model, criterion, optimizer, scheduler