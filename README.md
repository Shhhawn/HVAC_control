

- networks.py: 网络结构
- agent.py: 包含了最基础的 PPOAgent，没有拉格朗日法
- agent_sac.py: Soft Actor-Critic 算法
- agent_safe.py: 带有拉格朗日乘子的 PPOAgent


- run_pareto_search.py: 寻找 $w=10.0$ 的 Pareto 探索历史脚本
- run_extractor_ablation.py: 跑出 Vanilla -> Channel -> Gate -> Full 的消融实验脚本
- plot_extractor_ablation.py: 【画图脚本】把消融实验的数据画成图
- run_baseline_sac.py: 跑baseline脚本
- run_safe_ppo.py: 用于训练加入了拉格朗日函数的全量模型，当模型有改动时，需要重新跑一遍这个来获得参数
- run_generalization_test.py: 将训练好的参数拿去在混合气候中进行测试
- run_dcn_comparison.py: 比较DCN-v1和DCN-v2的效果