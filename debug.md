
#### 1. 地图路径问题
```from gfootball import``` 直接导向了```anaconda/lib/gfootball```，所以把scenarios文件复制过去就好了，只出现在dan的server上

#### 2. 临时缩小训练步骤

源代码中这两行是注释掉的，这里为了测试test加速child训练临时加上

```create_task.py lin 21-23```

data["env_args"]["rewards"] = 'scoring, reward_test'
data["t_max"] = 2000

#### 3. torch.load 报错
```torch.load(xxx)``` 在torch 2.6以后需要改成 ```torch.load(xxx, weight_only=False)```

#### 4. 为了测试debug
关闭并行环境启动，设置
```parallel_runner.py``` 中 line 18 为 ```self.batch_size = 1```

#### 5. default.yaml 修改:

原始为

save_model: False # Save the models to disk
save_model_interval: 50000 # Save models after this many timesteps

修改为

save_model: True # Save the models to disk
save_model_interval: 1000 # Save models after this many timesteps

#### 6. run.py 的改动（正式代码需要删除）
run.py line 237

新增一个指定500000变为2000 steps

#### 7. load actor/critic_init.th的改动
注意这里merge的policy是将子任务的参数平均，得到的共享参数，除非修改doe_controller，但那又涉及到role assignment的问题

#### 8. 固定长度one-hot的改动
ac.py line50, basic controller.py line89, doe_controller.py line115

#### TODO

run.py line 286 目前存储ckpt路径奇怪
save_path
'/data/qiaodan/projects/GRF_SUBTASK/doe_epymarl-main/results/models/ia2c_seed114514_scenario_layer2_decomposition0_subtask6_2025-05-05 11:45:34.649414/150'

line 215 目前并未成功load policy
因为ckpt path是 ""
而且需要解决team merged policy的问题，建议都存到decomposition/group文件夹下，但是分别命名为init policy和普通存储的final policy。
final policy用于在下一层target task训练时load，组合成init policy



doe的label和agent id要小心点，比如subtask 7 和 8 的合并顺序？
两个team都是0-N

注意 role list 作为MLP cls的train和load的label，检查这个label对超参数的影响


注意doe需要检查，预测的state对应label判断条件换成是否等于当前group id，因为train时候的label 替换成了group id为准，load doe时直接加载classifier nn params