import sys
import glob
import os
import shutil
import time
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.misc import load_checkpoint
import pointcept.utils.comm as comm
from pointcept.engines.test import TESTERS

from .default import HookBase
from .builder import HOOKS
from pathlib import Path

from numpy.core.multiarray import scalar
import torch

# allow the numpy scalar in weights_only loads
torch.serialization.add_safe_globals([scalar])


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        # MSC pretrain do not have offset information. Comment the code for support MSC
        # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
        #        "Scan {batch_size} ({points_num}) ".format(
        #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
        #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
        #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
        #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
        # )
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None → only ‘model_last’

    def after_epoch(self):
        # ---------------  COMPUTE (all ranks) ---------------
        # gather best‑metric bookkeeping exactly as before
        is_best = False
        if is_main_process() and self.trainer.cfg.evaluate:
            cur_val = self.trainer.comm_info["current_metric_value"]
            cur_name = self.trainer.comm_info["current_metric_name"]
            if cur_val > self.trainer.best_metric_value:
                self.trainer.best_metric_value = cur_val
                is_best = True
                self.trainer.logger.info(
                    f"Best validation {cur_name} updated to: {cur_val:.4f}"
                )
            self.trainer.logger.info(
                f"Currently Best {cur_name}: {self.trainer.best_metric_value:.4f}"
            )

        # make **sure every pending kernel is done** before anyone touches I/O
        torch.cuda.synchronize()
        synchronize()

        # ---------------  I/O (rank‑0 only) ---------------
        if is_main_process():
            ckpt_dir = Path(self.trainer.cfg.save_path) / "model"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            fname_last = ckpt_dir / "model_last.pth"

            self.trainer.logger.info(f"Saving checkpoint to: {fname_last}")
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                f"{fname_last}.tmp",
            )
            os.replace(f"{fname_last}.tmp", fname_last)

            if is_best:
                shutil.copyfile(fname_last, ckpt_dir / "model_best.pth")
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    fname_last, ckpt_dir / f"epoch_{self.trainer.epoch + 1}.pth"
                )

        # ---------------  RESUME TRAINING (all ranks) ---------------
        synchronize()  # wait until file rename finishes


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_eval(self):
        self.before_train()

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )

            # Get model state dict for shape validation
            model_state = self.trainer.model.state_dict()
            weight = OrderedDict()

            self.trainer.logger.info(
                f"Processing checkpoint keys with keyword: {self.keywords} -> {self.replacement}"
            )
            skipped_keys = {"shape_mismatch": [], "not_in_model": []}

            for key, value in checkpoint["state_dict"].items():
                # Process key for compatibility
                processed_key = key
                if not key.startswith("module."):
                    processed_key = "module." + key  # Add prefix for DDP format
                if self.keywords in processed_key:
                    processed_key = processed_key.replace(
                        self.keywords, self.replacement
                    )
                if comm.get_world_size() == 1:  # Remove prefix if not DDP
                    processed_key = (
                        processed_key[7:]
                        if processed_key.startswith("module.")
                        else processed_key
                    )

                # Key validation logic
                if processed_key in model_state:
                    if model_state[processed_key].shape == value.shape:
                        weight[processed_key] = value
                    else:
                        skipped_keys["shape_mismatch"].append(
                            f"{key} (ckpt shape: {value.shape} vs model shape: {model_state[processed_key].shape})"
                        )
                else:
                    skipped_keys["not_in_model"].append(processed_key)

            # Load filtered weights
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )

            # Detailed logging
            self.trainer.logger.info(
                f"Successfully loaded {len(weight)}/{len(checkpoint['state_dict'])} keys"
            )
            if skipped_keys["shape_mismatch"]:
                self.trainer.logger.warning(
                    f"Skipped {len(skipped_keys['shape_mismatch'])} keys due to shape mismatch:\n"
                )
            if skipped_keys["not_in_model"]:
                self.trainer.logger.warning(
                    f"Skipped {len(skipped_keys['not_in_model'])} keys not in model:\n"
                    f"{skipped_keys['not_in_model']}"
                )
            if self.strict:
                self.trainer.logger.info(
                    f"Strict mode: Missing keys: {load_state_info[0]}"
                )
                self.trainer.logger.info(
                    f"Strict mode: Unexpected keys: {load_state_info[1]}"
                )

            # Resume training if needed
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        # if cfg.test is a dict => single tester
        # if cfg.test is a list => multiple testers
        if isinstance(cfg.test, dict):
            tester = TESTERS.build(
                dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
            )  # e.g. test = dict(type='SemSegTester', verbose=True)
            if self.test_last:
                self.trainer.logger.info(
                    "=> Testing on model_last (current weight) ..."
                )
            else:
                best_path = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_best.pth"
                )
                self.trainer.logger.info("=> Testing on model_best...")
                checkpoint = torch.load(best_path)
                state_dict = checkpoint["state_dict"]
                tester.model.load_state_dict(state_dict, strict=True)
                self.trainer.logger.info(f"Loaded ckpt from {best_path}")
            tester.test()
        elif isinstance(cfg.test, list):
            for i, test_cfg in enumerate(cfg.test):
                tester = TESTERS.build(
                    dict(type=test_cfg.type, cfg=cfg, model=self.trainer.model, index=i)
                )
                if self.test_last:
                    self.trainer.logger.info(
                        "=> Testing on model_last (current weight) ..."
                    )
                else:
                    best_path = os.path.join(
                        self.trainer.cfg.save_path, "model", "model_best.pth"
                    )
                    self.trainer.logger.info("=> Testing on model_best...")
                    load_checkpoint(tester.model, best_path)
                    self.trainer.logger.info(f"Loaded ckpt from {best_path}")
                tester.test()
                del tester
                torch.cuda.empty_cache()


@HOOKS.register_module()
class BeginningEvaluator(HookBase):  # for testing
    def __init__(self, test_last=False):
        self.test_last = test_last

    def before_epoch(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>> Beginning Evaluation Before Training >>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        if isinstance(cfg.test, dict):
            tester = TESTERS.build(
                dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
            )  # e.g. test = dict(type='SemSegTester', verbose=True)
            if self.test_last:
                self.trainer.logger.info("=> Testing on model_last (current weight)...")
            else:
                self.trainer.logger.info("=> Testing on model_best ...")
                best_path = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_best.pth"
                )
                checkpoint = torch.load(best_path)
                self.trainer.logger.info(f"Loading ckpt from {best_path}")
                state_dict = checkpoint["state_dict"]
                tester.model.load_state_dict(state_dict, strict=True)
            tester.test()
        elif isinstance(cfg.test, list):
            for i, test_cfg in enumerate(cfg.test):
                tester = TESTERS.build(
                    dict(type=test_cfg.type, cfg=cfg, model=self.trainer.model, index=i)
                )
                if self.test_last:
                    self.trainer.logger.info(
                        "=> Testing on model_last (current weight)..."
                    )
                else:
                    self.trainer.logger.info("=> Testing on model_best ...")
                    best_path = os.path.join(
                        self.trainer.cfg.save_path, "model", "model_best.pth"
                    )
                    load_checkpoint(tester.model, best_path)
                    self.trainer.logger.info(f"Loading ckpt from {best_path}")
                tester.test()
                del tester
                torch.cuda.empty_cache()
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Beginning Evaluator, Skip Training >>>>>>>>>>>>>>>>"
        )
        sys.exit(0)


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)
