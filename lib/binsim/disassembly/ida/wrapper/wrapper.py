import os
import logging
import subprocess
import sys
import psutil
from binsim.disassembly.ida.utils import HeartbeatServer, HeartbeatClient

logger = logging.getLogger("IDAExeWrapper")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class IDACommandLine:
    def __init__(self, ida_path, script_path, target_file, ida_switches=None, ida_kwargs=None,
                 script_switch=None, script_kwargs=None, header=None):
        self.ida_path = ida_path
        self.script_path = script_path
        self.target_file = target_file
        self.ida_switches = ida_switches if ida_switches is not None else []
        self.ida_kwargs = ida_kwargs if ida_kwargs is not None else {}
        self.script_switch = script_switch if script_switch is not None else []
        self.script_kwargs = script_kwargs if script_kwargs is not None else {}
        self.header = header if header is not None else []

        assert os.path.exists(ida_path), f"IDA executable ({ida_path}) doesn't exist!"

    def __str__(self):
        ida_switches = " ".join(self.ida_switches)
        ida_kwargs = " ".join([f"{key} {value}" for key, value in self.ida_kwargs.items()])
        script_switch = " ".join(self.script_switch)
        script_kwargs = " ".join([f"{key} {value}" for key, value in self.script_kwargs.items()])
        header = " ".join(self.header)

        prefix = ''
        if self.ida_path.endswith(".exe") and sys.platform == 'linux':
            prefix = 'wine'

        return f"{header} {prefix} \"{self.ida_path}\" {ida_switches} {ida_kwargs} -S\"{self.script_path} {script_switch} {script_kwargs}\" {self.target_file}"

    def add_ida_switch(self, switch):
        self.ida_switches.append(switch)

    def add_ida_kwargs(self, key, value):
        self.ida_kwargs[key] = value

    def add_script_switch(self, switch):
        self.script_switch.append(switch)

    def add_script_kwargs(self, key, value):
        self.script_kwargs[key] = value

    def set_target_file(self, target_file):
        self.target_file = target_file

    def add_header(self, header):
        self.header.append(header)


class IDAExeWrapper:
    """
    This class implements these functions:
    1. Extract all functions ast from the specified binary file and save to the database
    2. Extract A specific function ast from the specified binary file and save to the database
    3. Batch version of function1: accessing all ELF files from the specified folder, extract all function asts and save to Database
    4. Read A specific ELF file from the specified folder, extract all the functions ast, save to the database
    """

    def __init__(self, ida_path, script_path, timeout, logfile=None, with_ui=False):
        self.Script = os.path.abspath(script_path)
        self._ida_path = ida_path
        self._timeout = timeout
        self._logfile = logfile
        self._with_ui = with_ui
        self.arch_support = ['powerpc', 'ppc', 'x86', 'x86-64', 'arm', "80386"]

    def extract_function_ast(self,
                             binary_path,
                             outfile,
                             db_file,
                             keep_thunk=False,
                             keep_unnamed=False,
                             keep_small=False,
                             small_ins_threshold=10,
                             small_graph_threshold=3,
                             keep_large=False,
                             large_ins_threshold=10000,
                             large_graph_threshold=500,
                             reanalysis=False,
                             regenerate=False,
                             **kwargs):  # the described function 2
        command = IDACommandLine(self._ida_path, self.Script, binary_path)
        command.add_script_kwargs("--database", outfile)

        if os.path.exists(outfile) and not reanalysis:
            return

        if keep_thunk:
            command.add_script_switch("--keep-thunk")
        if not keep_small:
            command.add_script_kwargs("--small-ins-threshold", small_ins_threshold)
            command.add_script_kwargs("--small-graph-threshold", small_graph_threshold)
        else:
            command.add_script_switch("--keep-small")

        if not keep_large:
            command.add_script_kwargs("--large-ins-threshold", large_ins_threshold)
            command.add_script_kwargs("--large-graph-threshold", large_graph_threshold)
        else:
            command.add_script_switch("--keep-large")

        if keep_unnamed:
            command.add_script_switch("--keep-unnamed")

        for key, value in kwargs.items():
            # Todo: concating the key and value directly may cause command injection
            key = key.replace("_", "-")
            if isinstance(value, str) and len(value) == 0:
                continue
            command.add_script_kwargs(f"--{key}", value)

        if self._logfile:
            command.add_script_kwargs("--logfile", self._logfile)

        if sys.platform == 'linux' and not self._with_ui:
            command.add_header("TVHEADLESS=1")

        if not self._with_ui:
            command.add_ida_switch("-A")
        command.add_ida_switch("-Lida.log")

        if db_file is not None:
            if regenerate or not os.path.exists(db_file):
                command.add_ida_switch("-c")
                if os.path.exists(db_file):
                    os.remove(db_file)
                command.add_ida_switch(f"-o\"{db_file}\"")
            else:
                command.target_file = db_file

        monitor_server = HeartbeatServer(timeout=self._timeout)
        p = self.wait_process(command, binary_path, monitor_server)
        if p is None:
            return

        if p.returncode != 0:
            raise RuntimeError(f"IDA Pro exited with error code {p.returncode}."
                               f"Please check the log file {self._logfile} for more details.")

    def wait_process(self, command, binary_path, monitor_server: HeartbeatServer):
        functions_to_skip = []
        max_retry_time = 5
        retry_time = 0
        while True:
            monitor_server.start(extra_data=functions_to_skip)
            command.add_script_kwargs('--monitor-port', monitor_server.port)
            command.add_script_kwargs('--monitor-host', monitor_server.host)
            cmd = str(command)
            proc = subprocess.Popen(cmd, shell=True)
            try:
                monitor_server.monitor(proc)
                proc.wait()
            except TimeoutError as e:
                logger.error(
                    f"[Error] Timeout while disassembling {binary_path}. Retrying...({retry_time}/{max_retry_time})")
                for child in psutil.Process(proc.pid).children(recursive=True):
                    child.kill()
                func_addr = monitor_server.last_message
                if func_addr is not None:
                    functions_to_skip.append(func_addr)
                retry_time += 1
                if max_retry_time < retry_time:
                    logger.error(f"[Error] Retry limit exceeded while disassembling {binary_path}.")
                    return None
                continue
            except RuntimeError as r:
                logger.error(f"[Error] RuntimeError while dealing with {binary_path}")
                return None
            except ValueError as e:
                logger.error(f"[Error] ValueError while dealing with {binary_path}")
                for child in psutil.Process(proc.pid).children(recursive=True):
                    child.kill()
                return None
            except Exception as e:
                logger.error(f"[Error] ValueError while dealing with {binary_path}")
                for child in psutil.Process(proc.pid).children(recursive=True):
                    child.kill()
                return None
            return proc


def execute_ida_script(ida_path,
                       script_path,
                       binary,
                       out_file,
                       db_file,
                       logfile=None,
                       timeout=120,
                       with_ui=False,
                       keep_thunk=False,
                       keep_unnamed=False,
                       keep_small=False,
                       small_ins_threshold=10,
                       small_graph_threshold=3,
                       keep_large=False,
                       large_ins_threshold=10000,
                       large_graph_threshold=500,
                       reanalysis=False,
                       regenerate=False,
                       **kwargs):
    assert binary, "binary or directory must be specified"
    out_file = os.path.abspath(out_file)
    ida_path = os.path.abspath(ida_path)
    if logfile is not None:
        logfile = os.path.abspath(logfile)
    db_file = os.path.abspath(db_file)
    assert os.path.exists(ida_path), "The provided ida_path is not a valid ida executable."
    assert logfile is None or not os.path.isdir(logfile), "The provided logfile must be a file, not a directory."
    assert not os.path.isdir(out_file), "The provided database must be a file, not a directory."
    assert not os.path.isdir(db_file), "The provided database must be a file, not a directory."
    wrapper = IDAExeWrapper(ida_path, timeout=timeout, with_ui=with_ui, logfile=logfile, script_path=script_path)
    binary = os.path.abspath(binary)
    wrapper.extract_function_ast(binary,
                                 outfile=out_file,
                                 db_file=db_file,
                                 keep_thunk=keep_thunk,
                                 keep_unnamed=keep_unnamed,
                                 keep_small=keep_small,
                                 small_ins_threshold=small_ins_threshold,
                                 small_graph_threshold=small_graph_threshold,
                                 keep_large=keep_large,
                                 large_ins_threshold=large_ins_threshold,
                                 large_graph_threshold=large_graph_threshold,
                                 regenerate=regenerate,
                                 reanalysis=reanalysis,
                                 **kwargs)
