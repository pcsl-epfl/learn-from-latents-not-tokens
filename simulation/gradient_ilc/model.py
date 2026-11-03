from __future__ import annotations

from ptc import add_ptc_args, init_ptc_from_args


def add_model_args(parent_parser):
    group = parent_parser.add_argument_group("Model Selection")
    group.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    group.add_argument(
        "--token_dimension",
        type=int,
        default=None,
        help="Token/cluster dimension for softmax outputs (defaults to rhm_v)",
    )
    group.add_argument(
        "--model_depth",
        type=int,
        default=None,
        help="Number of module pairs (defaults to rhm_L-1)",
    )
    group.add_argument(
        "--enable_probes",
        action="store_true",
        help="Enable linear probes on cluster outputs",
    )
    group.add_argument(
        "--stop_grad_between_modules",
        action="store_true",
        help="Stop gradients between stacked modules",
    )
    parent_parser = add_ptc_args(parent_parser)
    return parent_parser


def _apply_model_defaults(args):
    if getattr(args, "token_dimension", None) is None:
        args.token_dimension = args.rhm_v
    if getattr(args, "model_depth", None) is None:
        args.model_depth = max(0, int(args.rhm_L) - 1)
    return args


def init_model(args, resumption_state=None):
    _apply_model_defaults(args)
    if resumption_state is not None:
        model = init_model(args, resumption_state=None)
        model.load_state_dict(resumption_state["student"])
        model.to(args.device)
        return model
    if args.ptc:
        model = init_ptc_from_args(args)
    else:
        raise ValueError("No model type specified. Please set --ptc.")
    model.to(args.device)
    return model


def init_teacher(student, args, resumption_state=None):
    _apply_model_defaults(args)
    if resumption_state is not None:
        if resumption_state.get("teacher") is None:
            return None
        teacher = init_teacher(student, args, resumption_state=None)
        teacher.load_state_dict(resumption_state["teacher"])
        teacher = teacher.to(args.device)
        return teacher
    if args.ema_alpha > 0:
        teacher = init_model(args, resumption_state=None)
        teacher.load_state_dict(student.state_dict())
        teacher = teacher.to(args.device)
        return teacher
    return None
