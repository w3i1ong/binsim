import pickle
import tempfile
import sys, os, logging
from binsim.disassembly.utils import BinaryBase
from binsim.disassembly.utils import strip_file, get_architecture
from binsim.disassembly.core.disassembler import DisassemblerBase
from binsim.disassembly.backend.ida.wrapper import IDAExeWrapper

platform = sys.platform

logger = logging.getLogger("IDADisassembler")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class IDADisassembler(DisassemblerBase):
    def __init__(self, normalizer, ida_path: str, **kwargs):
        super().__init__(normalizer)
        self._ida_path = ida_path
        self._normalizer_kwargs = kwargs
        self.check_ida_path(ida_path)

    def check_ida_path(self, ida_path):
        assert os.path.isdir(ida_path), ("You must provide the root directory of IDA, not the path to ida.exe or ida64.exe."
                                         f"The provided path is {ida_path}.")
        if os.path.exists(os.path.join(ida_path, "ida64.exe")) or os.path.exists(os.path.join(ida_path, "ida64")):
            return
        raise ValueError(f"The provided path {ida_path} is not a valid IDA path.")

    def visit_functions(self, filename,
                        db_file=None,
                        with_ui=False,
                        load_pdb=False,
                        regenerate=False,
                        keep_thunks=False,
                        keep_unnamed=False,
                        verbose=True,
                        keep_large=False,
                        large_ins_threshold=10000,
                        large_graph_threshold=1000,
                        keep_small=False,
                        small_ins_threshold=10,
                        small_graph_threshold=5,
                        log_file=None,
                        timeout=15,
                        need_strip=True):
        # check arch
        try:
            arch = get_architecture(filename)
        except ValueError as e:
            logger.error(f"Cannot get arch of {filename}, skipped")
            return None
        bits = 64 if "64" in arch else 32
        # check database file
        if db_file is not None:
            if bits == 32 and not db_file.endswith(".idb"):
                db_file = f"{db_file}.idb"
            elif bits == 64 and not db_file.endswith(".i64"):
                db_file = f"{db_file}.i64"
        else:
            if bits == 32:
                db_file = tempfile.mktemp(".idb")
            else:
                db_file = tempfile.mktemp(".i64")
        kwargs = {
            "load_pdb": load_pdb, "keep_thunks": keep_thunks, "arch": arch,
            "keep_unnamed": keep_unnamed, "verbose": verbose, "keep_large": keep_large,
            "large_ins_threshold": large_ins_threshold, "large_graph_threshold": large_graph_threshold,
            "keep_small": keep_small, "small_ins_threshold": small_ins_threshold,
            "small_graph_threshold": small_graph_threshold
        }
        if os.path.exists(db_file) and not regenerate:
            filename = db_file
            db_file = None
        else:
            if need_strip:
                stripped_file = tempfile.mktemp(".strip")
                try:
                    strip_file(filename, stripped_file)
                except ValueError:
                    logger.error(f"Meet error while trying to strip {filename}")
                    return
                except RuntimeError:
                    logger.warning(f"Cannot strip {filename}. Just use the original file.")
                    stripped_file = filename

                symbols = BinaryBase(filename).addr2name
                filename = stripped_file
                kwargs["symbols"] = symbols

        out_file = tempfile.mktemp(".cfg")
        data = {
            "disassembler": self,
            "kwargs": kwargs,
            "outfile": out_file
        }

        # spwn ida process
        ida = self.get_proper_ida_exe(filename)
        cur_dir = os.path.split(__file__)[0]
        entry_script = os.path.join(cur_dir, "extra_scripts", "entry_point.py")
        ida = IDAExeWrapper(ida_path=ida, script_path=entry_script, timeout=timeout, with_ui=with_ui,
                            logfile=log_file)
        ida.extract_functions(filename, db_file=db_file, **data)
        if os.path.exists(out_file):
            with open(out_file, "rb") as f:
                cfgs = pickle.load(f)
            os.unlink(out_file)
            return cfgs
        return None


    def _visit_functions(self, arch,
                         load_pdb=None,
                         keep_thunks=False,
                         keep_unnamed=False,
                         verbose=True,
                         keep_large=False,
                         large_ins_threshold=1000,
                         large_graph_threshold=1000,
                         keep_small=False,
                         small_ins_threshold=10,
                         small_graph_threshold=10,
                         monitor_host="localhost",
                         monitor_port=5000,
                         need_decompile=False,
                         symbols:dict=None):
        import idautils, idc, idaapi
        from idaapi import FlowChart
        from idautils import FuncItems
        from binsim.disassembly.backend.ida.graph.utils import HeartbeatClient
        client = HeartbeatClient(monitor_host, monitor_port)

        if symbols is not None:
            for addr, func_name in symbols.items():
                idc.create_insn(addr)
                idc.add_func(addr)
                func = idaapi.get_func(addr)
                if func is not None and func.start_ea == addr:
                    idc.set_name(addr, func_name)
        print(f"{need_decompile:=}")
        idaapi.auto_wait()
        if need_decompile:
            idaapi.load_plugin_decompiler()

        normalizer = self.normalizer(arch, **self._normalizer_kwargs)
        cfg_list = []
        functions_to_skip = client.connect()
        # iterate each function
        for func in idautils.Functions():
            if func in functions_to_skip:
                continue
            client.ping(func)
            function = idaapi.get_func(func)
            func_name = idc.get_func_name(func)

            if not keep_thunks and (idc.get_func_flags(func) & idc.FUNC_THUNK):
                logger.info(f"Skip thunk function {func_name}, as it's a thunk function.")
                continue
            if idc.get_segm_name(func) in ['.plt', 'extern', '.init', '.fini']:
                logger.info(f"Skip function {func_name}, as it's in .plt, extern, .init, .fini.")
                continue
            if func_name.startswith('sub_') and not keep_unnamed:
                logger.info(f"Skip function {func_name}, as it's unnamed.")
                continue
            # check instruction number

            flow_charts = FlowChart(function)
            bb_num = flow_charts.size
            ins_num = len(list(FuncItems(function.start_ea)))

            if not keep_small and ins_num < small_ins_threshold:
                logger.info(f"Skip function {func_name}, as it's too small. It has {ins_num} instructions,"
                            f" while the threshold is {small_ins_threshold}")
                continue

            if not keep_large and ins_num > large_ins_threshold:
                logger.info(f"Skip function {func_name}, as it's too large. It has {ins_num} instructions,"
                            f" while the threshold is {large_ins_threshold}")
                continue

            if not keep_small and bb_num < small_graph_threshold:
                logger.info(f"Skip function {func_name}, as it's too small. It has {bb_num} nodes,"
                            f" while the threshold is {small_graph_threshold}")
                continue

            if not keep_large and bb_num > large_graph_threshold:
                logger.info(f"Skip function {func_name}, as it's too large. It has {bb_num} nodes,"
                            f" while the threshold is {large_graph_threshold}")
                continue
            cfg = normalizer(function)
            if cfg is None:
                continue
            cfg_list.append(cfg)
        client.end()
        return cfg_list


    def disassemble(self, filename,
                    db_file=None,
                    with_ui=False,
                    load_pdb=False,
                    regenerate=False,
                    keep_thunks=False,
                    keep_unnamed=False,
                    verbose=True,
                    keep_large=False,
                    large_ins_threshold=10000,
                    large_graph_threshold=1000,
                    keep_small=False,
                    small_ins_threshold=10,
                    small_graph_threshold=5,
                    log_file=None,
                    timeout=10,
                    **kwargs):
        # check database file
        if db_file is not None:
            db_file = os.path.abspath(db_file)
            assert not os.path.isdir(db_file), "The database file cannot be a directory."
            db_dir = os.path.split(db_file)[0]
            os.makedirs(db_dir, exist_ok=True)

        if log_file is not None:
            assert log_file.endswith('.log'), "The log file must be a log file, not a directory."
            log_dir = os.path.split(log_file)[0]
            if log_dir != '' and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

        cfgs = self.visit_functions(filename, db_file=db_file, load_pdb=load_pdb, regenerate=regenerate,
                                    keep_thunks=keep_thunks, keep_unnamed=keep_unnamed, verbose=verbose,
                                    keep_large=keep_large, large_graph_threshold=large_graph_threshold,
                                    large_ins_threshold=large_ins_threshold, keep_small=keep_small,
                                    small_ins_threshold=small_ins_threshold, small_graph_threshold=small_graph_threshold,
                                    with_ui=with_ui, log_file=log_file, timeout=timeout)
        return cfgs


    def get_proper_ida_exe(self, filename):
        postfix = ""
        if os.path.exists(f"{self._ida_path}/ida64.exe"):
            postfix = ".exe"

        # for database file
        if filename.endswith("idb"):
            bits = 32
        elif filename.endswith("i64"):
            bits = 64
        else:
            # for binary file
            with open(filename, 'rb') as f:
                data = f.read(16)
            if data.startswith(b'\x7fELF'):
                if data[4] == 1:
                    bits = 32
                elif data[4] == 2:
                    bits = 64
                else:
                    raise ValueError("The provided file is neither a 32-bit ELF file nor a 64-bit ELF file.")
            else:
                raise ValueError("The provided file is neither an ELF file nor an IDA database file.")
        # For high version of IDA, there is no ida.exe, only ida64.exe, se we also need to check the existence of ida.exe.
        if bits == 32 and os.path.exists(f"{self._ida_path}/ida{postfix}"):
            return f"{self._ida_path}/ida{postfix}"
        else:
            return f"{self._ida_path}/ida64{postfix}"
