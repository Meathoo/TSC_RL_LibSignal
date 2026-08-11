import subprocess
import shlex
import sys
# 定義所有要跑的指令清單
tasks = [
    # "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow4x4 --prefix hyperlight_chunked_c8_seed0 --seed 0 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    # "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow4x4 --prefix hyperlight_chunked_c8_seed1 --seed 1 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    # "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow4x4 --prefix hyperlight_chunked_c8_seed2 --seed 2 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    # "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperlight_chunked_c8_seed0 --seed 0 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperlight_chunked_c8_seed1 --seed 1 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    "python3 run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperlight_chunked_c8_seed2 --seed 2 --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16",
    # hyperMAPPO learned64 FiLM
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow4x4  --prefix hyperlight_film_both_mlp_seed0  --seed 0  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow4x4  --prefix hyperlight_film_both_mlp_seed1  --seed 1  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow4x4  --prefix hyperlight_film_both_mlp_seed2  --seed 2  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow16x3  --prefix hyperlight_film_both_mlp_seed0  --seed 0  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow16x3  --prefix hyperlight_film_both_mlp_seed1  --seed 1  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow16x3  --prefix hyperlight_film_both_mlp_seed2  --seed 2  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow7x28  --prefix hyperlight_film_both_mlp_seed0  --seed 0  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow7x28  --prefix hyperlight_film_both_mlp_seed1  --seed 1  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # "python -u run.py  --task tsc  --agent hyperlight_mappo  --world cityflow  --network cityflow7x28  --prefix hyperlight_film_both_mlp_seed2  --seed 2  --ngpu 0  --agent_embedding_mode learned  --hyper_actor_arch mlp  --hyper_adapter_mode film  --hyper_critic_adapter_mode film  --hyper_film_scale 0.1  --hyper_residual False  --episodes 250  --profile_performance True",
    # 補完7x28的native_mappo learned64兩Seeds
    # 使用 seed0 hyperparameters.json 作完整 config snapshot，完成後自動整理到
    # cityflow_native_mappo/cityflow7x28/queue+phase+learned64(id)/seedX_learned64
    # "python -u batch_run_native_mappo_learned_7x28.py",

    # HyperLight Shared MLP
    # actor: queue + phase；actor 乾淨 MLP, learned ID 僅供 HyperLight critic 使用
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",

    # Parameter-matched HyperLight Shared MLP
    # actor: 32 -> 124 -> 116 -> 8，共 19,528 parameters
    # learned ID 僅供相同的 HyperLight centralized critic 使用
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_pm19528_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_actor_hidden1 124 --hyper_actor_hidden2 116 --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_pm19528_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_actor_hidden1 124 --hyper_actor_hidden2 116 --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_mlp_pm19528_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_actor_hidden1 124 --hyper_actor_hidden2 116 --hyper_adapter_mode none --hyper_residual False --episodes 250 --profile_performance True",

    # # HyperLight-FiLM MLP
    # # actor state: queue+phase；learned64 ID -> FiLM , critic 也有 ID hypernetwork
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperIRU_film_mlp_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode film --hyper_film_scale 0.1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_film_mlp_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode film --hyper_film_scale 0.1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_film_mlp_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode film --hyper_film_scale 0.1 --hyper_residual False --episodes 250 --profile_performance True",

    # # HyperLight Shared IRU n1
    # # actor state: queue+phase；actor IRU 且無 ID 和 modulation，ID 僅用於 critic
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperIRU_shared_iru1_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode none --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_iru1_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode none --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperIRU_shared_iru1_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode none --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",

    # HyperLight-FiLM IRU n1
    # actor state: queue+phase；learned64 ID -> IRU FiLM
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperIRU_film_iru1_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode film --hyper_film_scale 0.1 --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperIRU_film_iru1_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode film --hyper_film_scale 0.1 --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow7x28 --prefix hyperIRU_film_iru1_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch iru --hyper_adapter_mode film --hyper_film_scale 0.1 --iru_actor_steps 1 --iru_hidden_dim 64 --iru_num_blocks 1 --hyper_residual False --episodes 250 --profile_performance True",



    # mappo_iru(n=1) actor only phase+queue+learned64(id)
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow16x3 --prefix seed1_actor_iru1_learned_ep100 --seed 1 --ngpu 0 --native_actor_arch iru --native_value_arch mlp --iru_steps 1 --native_use_agent_id True --native_agent_id_mode learned --profile_performance true",
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow16x3 --prefix seed2_actor_iru1_learned_ep100 --seed 2 --ngpu 0 --native_actor_arch iru --native_value_arch mlp --iru_steps 1 --native_use_agent_id True --native_agent_id_mode learned --profile_performance true",    

    # mappo_iru(n=5) actor only phase+queue+learned64(id)
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow16x3 --prefix seed1_actor_iru5_learned_ep100 --seed 1 --ngpu 0 --native_actor_arch iru --native_value_arch mlp --iru_steps 5 --native_use_agent_id True --native_agent_id_mode learned --profile_performance true",
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow16x3 --prefix seed2_actor_iru5_learned_ep100 --seed 2 --ngpu 0 --native_actor_arch iru --native_value_arch mlp --iru_steps 5 --native_use_agent_id True --native_agent_id_mode learned --profile_performance true",

    # mappo_iru(n=5) both actor and value phase+queue+learned64(id)
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow16x3 --prefix seed0_both_iru5_learned_ep100 --seed 0 --ngpu 0 --native_actor_arch iru --native_value_arch iru --iru_steps 5 --native_use_agent_id True --native_agent_id_mode learned --profile_performance true",

    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow16x3 --prefix seed0_graph_mappo --seed 0 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow16x3 --prefix seed1_graph_mappo --seed 1 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow16x3 --prefix seed2_graph_mappo --seed 2 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow7x28 --prefix seed0_graph_mappo --seed 0 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow7x28 --prefix seed1_graph_mappo --seed 1 --ngpu 0",
    # "python -u run.py --task tsc --agent hyperlight_graph_mappo --world cityflow --network cityflow7x28 --prefix seed2_graph_mappo --seed 2 --ngpu 0",
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow7x28 --prefix seed0_iru1_noID --seed 0 --ngpu 0 --native_actor_arch iru --native_value_arch iru --iru_steps 1",
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow7x28 --prefix seed0_iru2_noID --seed 0 --ngpu 0 --native_actor_arch iru --native_value_arch iru --iru_steps 2",
    # "python -u run.py --task tsc --agent mappo_iru --world cityflow --network cityflow7x28 --prefix seed0_iru5_noID --seed 0 --ngpu 0 --native_actor_arch iru --native_value_arch iru --iru_steps 5",

    # mappo queue+phase
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix queue+phase_seed0 --seed 0 --ngpu 0 --native_use_agent_id False --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix queue+phase_seed1 --seed 1 --ngpu 0 --native_use_agent_id False --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix queue+phase_seed2 --seed 2 --ngpu 0 --native_use_agent_id False --profile_performance true",
    # mappo queue+phase+onehot(id)
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed0 --seed 0 --ngpu 0 --native_use_agent_id True --native_agent_id_mode one_hot --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed1 --seed 1 --ngpu 0 --native_use_agent_id True --native_agent_id_mode one_hot --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed2 --seed 2 --ngpu 0 --native_use_agent_id True --native_agent_id_mode one_hot --profile_performance true",
    # mappo queue+phase+learned64(id)
    # "python -u run.py --task tsc --agent native_mappo_learned --world cityflow --network cityflow16x3 --prefix learned64_seed0 --seed 0 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo_learned --world cityflow --network cityflow16x3 --prefix learned64_seed1 --seed 1 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent native_mappo_learned --world cityflow --network cityflow16x3 --prefix learned64_seed2 --seed 2 --ngpu 0 --profile_performance true",

    # mappo queue+phase+onehot(id)
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed0 --agent_embedding_mode one_hot --seed 0 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed1 --agent_embedding_mode one_hot --seed 1 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix onehot_seed2 --agent_embedding_mode one_hot --seed 2 --ngpu 0 --profile_performance true",
    # mappo queue+phase+learned64(id)
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix learned64_seed0 --agent_embedding_mode learned --seed 0 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix learned64_seed1 --agent_embedding_mode learned --seed 1 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix learned64_seed2 --agent_embedding_mode learned --seed 2 --ngpu 0 --profile_performance true",
    # mappo queue+phase+learned64(id)+residual
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix res002_seed0 --hyper_residual True --seed 0 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix res002_seed1 --hyper_residual True --seed 1 --ngpu 0 --profile_performance true",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix res002_seed2 --hyper_residual True --seed 2 --ngpu 0 --profile_performance true",
    
    # Pure HyperLight MAPPO + learned ID head residual
    # Shared MLP/value base；hypernetwork 只產生 actor/value output-head delta
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperlight_head_res002_seed0 --seed 0 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode generated --hyper_residual True --hyper_residual_mode head --hyper_residual_scale 0.02 --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperlight_head_res002_seed1 --seed 1 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode generated --hyper_residual True --hyper_residual_mode head --hyper_residual_scale 0.02 --episodes 250 --profile_performance True",
    # "python -u run.py --task tsc --agent hyperlight_mappo --world cityflow --network cityflow16x3 --prefix hyperlight_head_res002_seed2 --seed 2 --ngpu 0 --agent_embedding_mode learned --hyper_actor_arch mlp --hyper_adapter_mode generated --hyper_residual True --hyper_residual_mode head --hyper_residual_scale 0.02 --episodes 250 --profile_performance True",

]

for cmd in tasks:
    print(f"正在執行: {cmd}")
    # shell=True 允許直接執行字串指令
    argv = shlex.split(cmd)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    print(f"正在執行: {shlex.join(argv)}", flush=True)
    subprocess.run(argv, check=True)

print("所有實驗已完成！")
