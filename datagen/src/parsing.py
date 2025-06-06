import os
import sys
import logging
logger = logging.getLogger(__name__)
import yaml
from stability_analysis.data import get_data_path


def parse_setup_file(setup_path):
    # Check setup file path
    if not setup_path:
        current_directory = os.path.dirname(__file__)
        setup_path = os.path.join(current_directory,
                                  "../../setup/default_setup.yaml")
        logger.warning(f"Setup file not specified. Using default setup file: "
                f"{setup_path}")
    else:
        logger.info(f"Setup file: {setup_path}")
    if not os.path.exists(setup_path):
        logging.error(f"Setup file {setup_path} not found")
        raise FileNotFoundError(f"Setup file {setup_path} not found")

    # Load case parameters
    return load_yaml(setup_path)



def parse_args(argv):
    working_dir = None
    path_data = None
    setup_path = None

    args = argv[1:]
    # Do not mix flagged arguments with non-flagged arguments
    i = 0
    use_flag_args = True
    while args:
        arg = args.pop(0)
        if arg.startswith('--working_dir='):
            working_dir = arg.split('=', 1)[1]
        elif arg.startswith('--path_data='):
            path_data = arg.split('=', 1)[1]
        elif arg.startswith('--setup='):
            setup_path = arg.split('=', 1)[1]
        else:
            use_flag_args = False
            if i == 0:
                working_dir = arg
            elif i == 1:
                path_data = arg
            elif i == 2:
                setup_path = arg
        i += 1
    if not use_flag_args:
        logger.info("Using arguments without flags")

    # Check paths
    if not working_dir:
        working_dir = ""
        logger.warning(f"Working directory not specified. Using current directory: "
              f"{os.getcwd()}")
    else:
        if not os.path.exists(working_dir):
            message = f"Working directory {working_dir} not found"
            logger.error(message)
            raise FileNotFoundError(message)
        else:
            logger.info("Working directory:", working_dir)
    if not path_data:
        path_data = get_data_path()
        logger.warning(f"Path data not specified. Using default path: {path_data}")
    else:
        if not os.path.exists(path_data):
            raise FileNotFoundError(f"Path data {path_data} not found")
        else:
            logger.info("Path data:", path_data)

    return working_dir, path_data, setup_path


def load_yaml(content):
    try:
        content = os.path.expanduser(content)
        # Try to interpret content as a file path
        with open(content, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        try:
            # If not a file, try to interpret content as a YAML string
            return yaml.safe_load(content)
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML content: {exc}")
            sys.exit(1)
