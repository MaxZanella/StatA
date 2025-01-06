import os
import torch
import copy
text_only_methods = ["coop", "prograd", "taskres"]
logits_only_methods = ["cocoop", "tip_adapter_f"]
logits_and_vision_features_methods = ["maple", "promptsrc", "plot"]

eva_clip = ["EVA-CLIP-8B"]

def prepare_for_bench(args, test_features, test_labels, clip_weights):
    if args.prototypes_dataset == 'fgvc':
        args.prototypes_dataset = 'fgvc_aircraft'

    fuse = False
    if args.setting == 'Domain-generalization':
        fuse=True

    initial_prototypes = None
    initial_logits = None

    clip_proto = clip_weights

    initial_features = test_features
    initial_labels = test_labels

    if args.prototypes_method in eva_clip:
        proto_path = os.path.join(args.prototypes_path, "EVA-CLIP", args.prototypes_dataset, args.prototypes_method, "text_features.pt")
        features_path = os.path.join(args.prototypes_path, "EVA-CLIP", args.prototypes_dataset, args.prototypes_method, "image_features.pt")

        initial_prototypes = torch.load(proto_path).T.detach().requires_grad_(False)
        initial_features = torch.load(features_path).detach().requires_grad_(False)
        clip_proto = initial_prototypes

    elif args.prototypes_method in text_only_methods:
        proto_path = os.path.join(args.prototypes_path, args.setting, args.prototypes_method, args.backbone,
                                  "{}shots".format(args.prototypes_shots), args.prototypes_dataset,
                                  "seed{}".format(args.seed), "text_features.pt")
        initial_prototypes = torch.load(proto_path).T.detach().requires_grad_(False)
    elif args.prototypes_method in logits_only_methods:
        logits_path = os.path.join(args.prototypes_path, args.setting, args.prototypes_method, args.backbone,
                                   "{}shots".format(args.prototypes_shots), args.prototypes_dataset,
                                   "seed{}".format(args.seed), "logits.pt")
        initial_logits = torch.load(logits_path).detach().requires_grad_(False)
    elif args.prototypes_method in logits_and_vision_features_methods:

        logits_path = os.path.join(args.prototypes_path, args.setting, args.prototypes_method, args.backbone,
                                   "{}shots".format(args.prototypes_shots), args.prototypes_dataset,
                                   "seed{}".format(args.seed), "logits.pt")

        initial_logits = torch.load(logits_path).detach().requires_grad_(False)

    if fuse:
        # change clip prototypes for the mapping
        dataset = args.dataset

        if dataset == "imagenet_a":  # K=200
            from datasets.imagenet_a import imagenet_a_mask
            mask = copy.copy(imagenet_a_mask)
            mask = torch.tensor(mask)
            initial_prototypes = initial_prototypes[:, mask.detach()]
        elif dataset == "imagenet_v2":  # K=200
            from datasets.imagenet_v2 import imagenet_v_mask
            mask = copy.copy(imagenet_v_mask)
            initial_prototypes = initial_prototypes[:, mask]
        elif dataset == "imagenet_r":  # K=200
            from datasets.imagenet_r import imagenet_r_mask
            mask = [i for i, m in enumerate(imagenet_r_mask) if m]
            initial_prototypes = initial_prototypes[:, mask]
        elif dataset == "imagenet_sketch":  # K=1000
            pass

    return initial_features, initial_labels, initial_prototypes, initial_logits, clip_proto
