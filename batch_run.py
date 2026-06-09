import subprocess

# 定義所有要跑的指令清單
tasks = [
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow4x4 --prefix seed0_rewardQueuePress --seed 0 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow6x6_bi --prefix seed0_rewardQueuePress --seed 0 --ngpu 0",
    "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix seed2_learned64_mlp_queuePress02_ep250 --seed 2 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_maspo --world cityflow --network cityflow7x28 --prefix seed0_learned64_mlp_queue_ep250_lr00005_epsilon0.3 --seed 0 --ngpu 0",
]

for cmd in tasks:
    print(f"正在執行: {cmd}")
    # shell=True 允許直接執行字串指令
    subprocess.run(cmd, shell=True, check=True)

print("所有實驗已完成！")