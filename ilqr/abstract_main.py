import logging
import logging.config
import os
from time import strftime, localtime

from omegaconf import OmegaConf


class AbstractMain:

    def __init__(self, overrides=None):
        self.log = None
        self.args, self.cfg = self.get_args_and_cfg(overrides)

        if not hasattr(self.args, 'verbose'):
            verbose = False
        else:
            verbose = self.args.verbose
        self.configure_log(self.cfg, verbose)
        if overrides is not None:
            self.log.info("=========================================")
            self.log.info(f"Overrides:\n{overrides.pretty()}")
        self.start()

    def start(self):
        raise NotImplementedError("Subclass must implement")

    def get_args(self):
        raise NotImplementedError("Subclass must implement")

    def get_args_and_cfg(self, overrides=None):
        args = self.get_args()
        cfg = OmegaConf.load(args.config)
        if hasattr(args, 'log_config') and args.log_config is not None:
            cfg.log_config = args.log_config
        if overrides is not None:
            cfg = cfg.merge(cfg, overrides)
        return args, cfg

    def configure_log(self, cfg, verbose=None):
        """
        :param cfg:
        :param verbose: all or root to activate verbose logging for all modules, otherwise a comma separated list of modules
        :return:
        """
        # configure target directory for all logs files (binary, text. models etc)
        log_dir_suffix = cfg.log_dir_suffix or strftime("%Y-%m-%d_%H-%M-%S", localtime())
        log_dir = os.path.join(cfg.log_dir or "logs", log_dir_suffix)
        cfg.full_log_dir = log_dir
        os.makedirs(cfg.full_log_dir, exist_ok=True)

        # using ruamel.yaml directly. OmegaConf is not an actual dict and it's confusing python logging config.
        from ruamel import yaml
        import io
        logfile = os.path.abspath(cfg.log_config)
        with io.open(logfile, 'r') as file:
            logcfg = yaml.safe_load(file)
            log_name = logcfg['handlers']['file']['filename']
            if not os.path.isabs(log_name):
                logcfg['handlers']['file']['filename'] = os.path.join(cfg.full_log_dir, log_name)
            logging.config.dictConfig(logcfg)

        # set log for this file only after configuring logging system
        self.log = logging.getLogger(__name__)

        if verbose:
            if verbose in ('all', 'root'):
                logging.getLogger().setLevel(logging.DEBUG)
                verbose = 'root'
            for logger in verbose.split(','):
                logging.getLogger(logger).setLevel(logging.DEBUG)

        self.log.info(f"Saving logs to {cfg.full_log_dir}")
