import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    ############################ MODE ARGS ############################
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        help='Options: train, test, generate, robust')
    
    return parser

def _args_as_dict(args):
    return vars(args)

def get_args(as_dict=False) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_known_args()[0]
    return _args_as_dict(args) if as_dict else args
