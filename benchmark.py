# Benchmark script for LightGlue on real images
#这个代码是用来做一些对比实验的
import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

torch.set_grad_enabled(False)


def measure(matcher, data, device="cuda", r=100):
    timings = np.zeros((r, 1))
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    # warmup
    for _ in range(10):
        _ = matcher(data)
    # measurements
    with torch.no_grad():
        for rep in range(r):
            if device.type == "cuda":
                starter.record()
                _ = matcher(data)
                ender.record()
                # sync gpu
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                _ = matcher(data)
                curr_time = (time.perf_counter() - start) * 1e3
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / r
    std_syn = np.std(timings)
    return {"mean": mean_syn, "std": std_syn}


def print_as_table(d, title, cnames):
    print()
    header = f"{title:30} " + " ".join([f"{x:>7}" for x in cnames])
    print(header)
    print("-" * len(header))
    for k, l in d.items():
        print(f"{k:30}", " ".join([f"{x:>7.1f}" for x in l]))


if __name__ == "__main__":
    # 设置解析器并添加描述信息
    parser = argparse.ArgumentParser(description="LightGlue的基准测试脚本")
    
    # 添加命令行参数
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="要进行基准测试的设备"
    )
    parser.add_argument("--compile", action="store_true", help="编译LightGlue运行")
    parser.add_argument(
        "--no_flash", action="store_true", help="禁用FlashAttention"
    )
    parser.add_argument(
        "--no_prune_thresholds",
        action="store_true",
        help="禁用修剪阈值（即始终进行修剪）"
    )
    parser.add_argument(
        "--add_superglue",
        action="store_true",
        help="将SuperGlue添加到基准测试中（需要hloc）"
    )
    parser.add_argument(
        "--measure", default="time", choices=["time", "log-time", "throughput"]
    )
    # 这里的repeat参数是用来设置测量重复次数的
    parser.add_argument(
        "--repeat", "--r", type=int, default=100, help="测量重复次数"
    )
    parser.add_argument(
        "--num_keypoints",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048, 4096],
        help="关键点的数量（以空格分隔的列表）"
    )
    parser.add_argument(
        "--matmul_precision", default="highest", choices=["highest", "high", "medium"]
    )
    parser.add_argument(
        "--save", default="output/campare.png", type=str, help="保存图形的路径"
    )
    
    # 解析命令行参数
    args = parser.parse_intermixed_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    print("Running benchmark on device:", device)

    # 设置图像文件路径
    images = Path("assets")
    # 定义输入图片字典，包含两个类别的图像名称和对应的加载图像结果
    inputs = {
        "easy": (
            load_image(images / "DSC_0411.JPG"),
            load_image(images / "DSC_0410.JPG"),
        ),
        "difficult": (
            load_image(images / "sacre_coeur1.jpg"),
            load_image(images / "sacre_coeur2.jpg"),
        ),
    }
    
    # 配置模型参数字典
    configs = {
        #"LightGlue-full": {
        "LightGlue": {
            "depth_confidence": -1,
            "width_confidence": -1,
        },
        # 'LG-prune': {
        #     'width_confidence': -1,
        # },
        # 'LG-depth': {
        #     'depth_confidence': -1,
        # },
        #"LightGlue-adaptive": {},
    }
    
    # 若存在编译参数，则为每个配置项添加对应的编译配置
    if args.compile:
        configs = {**configs, **{k + "-compile": v for k, v in configs.items()}}
    
    # SupeerGlue匹配模型参数字典
    sg_configs = {
        # 'SuperGlue': {},
        
        "SuperGlue-fast": {"sinkhorn_iterations": 5}
    }
    
    # 设置张量运算精度
    torch.set_float32_matmul_precision(args.matmul_precision)
    
    # 初始化结果字典，包含每个输入类别对应的默认空列表
    results = {k: defaultdict(list) for k, v in inputs.items()}

    # 初始化SuperPoint特征提取器
    extractor = SuperPoint(max_num_keypoints=None, detection_threshold=-1)
    extractor = extractor.eval().to(device)

    # 绘制结果图
    figsize = (len(inputs) * 4.5, 4.5)
    fig, axes = plt.subplots(1, len(inputs), sharey=True, figsize=figsize)
    axes = axes if len(inputs) > 1 else [axes]
    fig.canvas.manager.set_window_title(f"LightGlue benchmark ({device.type})")

    # 设置图形属性
    for title, ax in zip(inputs.keys(), axes):
        ax.set_xscale("log", base=2)
        
        bases = [2**x for x in range(7, 16)]
        # 设置坐标轴
        ax.set_xticks(bases, bases)
        ax.grid(which="major")
        if args.measure == "log-time":
            ax.set_yscale("log")
            yticks = [10**x for x in range(6)]
            ax.set_yticks(yticks, yticks)
            mpos = [10**x * i for x in range(6) for i in range(2, 10)]
            mlabel = [
                10**x * i if i in [2, 5] else None
                for x in range(6)
                for i in range(2, 10)
            ]
            ax.set_yticks(mpos, mlabel, minor=True)
            ax.grid(which="minor", linewidth=0.2)
        ax.set_title(title)

        # 设置图例
        ax.set_xlabel("# keypoints")
        if args.measure == "throughput":
            ax.set_ylabel("Throughput [pairs/s]")
        else:
            ax.set_ylabel("Latency [ms]")

    # 开始基准测试
    for name, conf in configs.items():
        print("Run benchmark for:", name)
        torch.cuda.empty_cache()
        # 初始化LightGlue匹配器
        matcher = LightGlue(features="superpoint", flash=not args.no_flash, **conf)
        if args.no_prune_thresholds:
            matcher.pruning_keypoint_thresholds = {
                k: -1 for k in matcher.pruning_keypoint_thresholds
            }
        # 加载SuperPoint特征提取器
        matcher = matcher.eval().to(device)
        if name.endswith("compile"):
            import torch._dynamo

            torch._dynamo.reset()  # avoid buffer overflow
            matcher.compile()
        # 开始测量
        for pair_name, ax in zip(inputs.keys(), axes):
            image0, image1 = [x.to(device) for x in inputs[pair_name]]
            runtimes = []
            for num_kpts in args.num_keypoints:
                extractor.conf.max_num_keypoints = num_kpts
                feats0 = extractor.extract(image0)
                feats1 = extractor.extract(image1)
                runtime = measure(
                    matcher,
                    {"image0": feats0, "image1": feats1},
                    device=device,
                    r=args.repeat,
                )["mean"]
        
                results[pair_name][name].append(
                    1000 / runtime if args.measure == "throughput" else runtime
                )
            ax.plot(
                args.num_keypoints, results[pair_name][name], label=name, marker="o"
            )
        del matcher, feats0, feats1

#    if args.add_superglue:
    if True:
        from hloc.matchers.superglue import SuperGlue
        # 开始SuperGlue基准测试
        for name, conf in sg_configs.items():
            print("Run benchmark for:", name)
            # 初始化SuperGlue匹配器
            matcher = SuperGlue(conf)
            # 加载SuperPoint特征提取器
            matcher = matcher.eval().to(device)
            # 开始测量
            # 这里的输入数据格式与LightGlue不同，需要进行转换
            for pair_name, ax in zip(inputs.keys(), axes):
                # 加载图像数据
                image0, image1 = [x.to(device) for x in inputs[pair_name]]
                runtimes = []
                for num_kpts in args.num_keypoints:
                    extractor.conf.max_num_keypoints = num_kpts
                    # 提取特征
                    feats0 = extractor.extract(image0)
                    feats1 = extractor.extract(image1)
                    # 转换输入数据格式
                    data = {
                        "image0": image0[None],
                        "image1": image1[None],
                        **{k + "0": v for k, v in feats0.items()},
                        **{k + "1": v for k, v in feats1.items()},
                    }
                    # 开始测量
                    data["scores0"] = data["keypoint_scores0"]
                    data["scores1"] = data["keypoint_scores1"]
                    # 转换数据格式
                    data["descriptors0"] = (
                        data["descriptors0"].transpose(-1, -2).contiguous()
                    )
                    data["descriptors1"] = (
                        data["descriptors1"].transpose(-1, -2).contiguous()
                    )
                    # 开始测量
                    runtime = measure(matcher, data, device=device, r=args.repeat)[
                        "mean"
                    ]
                    # 保存结果
                    results[pair_name][name].append(
                        1000 / runtime if args.measure == "throughput" else runtime
                    )
                ax.plot(
                    args.num_keypoints, results[pair_name][name], label=name, marker="o"
                )
            del matcher, data, image0, image1, feats0, feats1

    for name, runtimes in results.items():
        print_as_table(runtimes, name, args.num_keypoints)

    axes[0].legend()
    fig.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=fig.dpi)
    plt.show()
