import subprocess

# 定義所有要跑的指令清單
tasks = [
    "python run.py --task tsc --agent native_mappo --world cityflow --network cityflow4x4 --prefix seed0 --seed 0 --ngpu 0",
    "python run.py --task tsc --agent native_ppo --world cityflow --network cityflow6x6_bi --prefix seed0 --seed 0 --ngpu 0",
    "python run.py --task tsc --agent native_mappo --world cityflow --network cityflow6x6_bi --prefix seed0 --seed 0 --ngpu 0",
    "python run.py --task tsc --agent native_ppo --world cityflow --network cityflow7x28 --prefix seed0 --seed 0 --ngpu 0",
    "python run.py --task tsc --agent native_mappo --world cityflow --network cityflow7x28 --prefix seed0 --seed 0 --ngpu 0",
]

for cmd in tasks:
    print(f"正在執行: {cmd}")
    # shell=True 允許直接執行字串指令
    subprocess.run(cmd, shell=True, check=True)

print("所有實驗已完成！")