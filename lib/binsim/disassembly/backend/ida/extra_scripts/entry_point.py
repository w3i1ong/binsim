import idc
import logging
import pickle

logger = logging.getLogger("GEN-AST")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter("%(filename)s : %(message)s"))
logger.handlers[0].setLevel(logging.ERROR)


def init_file_handler(filename):
    global logger
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s : %(message)s"))
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kwargs", type=str, required=True, help="The file that save the kwargs.")
    ap.add_argument("--monitor-host", required=True, type=str, help="the host of the heartbeat server.")
    ap.add_argument("--monitor-port", required=True, type=int, help="the port of the heartbeat server.")
    args = ap.parse_args(idc.ARGV[1:])
    host, port = args.monitor_host, args.monitor_port
    # if args.logfile:
    #     init_file_handler(args.logfile)

    with open(args.kwargs, 'rb') as f:
        data = pickle.load(f)
    kwargs, out_file = data["kwargs"], data["outfile"]
    disassembler = data["disassembler"]
    idc.auto_wait()
    cfgs = disassembler._visit_functions(monitor_host=args.monitor_host, monitor_port=args.monitor_port, **kwargs)
    with open(out_file, 'wb') as f:
        pickle.dump(cfgs, f)
    idc.qexit(0)
