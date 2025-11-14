import numpy as np
import pickle
import json


class ChainballOptimalsGenerator:
    """
    根据论文描述生成Chainball-11环境的optimals配置

    论文规则：
    1. 每个状态有一个最优联合动作，前向概率为0.8
    2. 其他动作的前向概率从[0, 0.5]均匀采样
    3. 主任务(4 agents)：状态5只依赖(a1,a3)，状态7只依赖(a2,a4)
    4. 防守任务(2 agents)：状态1-4的最优动作与主任务的agent 1,2一致
    5. 进攻任务(2 agents)：状态8-11的最优动作与主任务的agent 3,4一致
    """

    def __init__(self, seed=1):
        self.seed = seed
        np.random.seed(seed)

    def generate_main_task_optimals(self, num_agents=4, chain_length=11):
        """
        生成主任务(Chainball-11)的optimals

        Returns:
            list: 长度为chain_length+2的列表，每个元素是num_agents个动作的元组
        """
        optimals = []

        for state in range(chain_length + 2):  # 0到12
            if state == 5:
                # 状态5：只依赖agent 1和3，agent 2和4用None
                a1 = np.random.randint(0, 4)
                a3 = np.random.randint(0, 4)
                opt_action = (a1, None, a3, None)

            elif state == 7:
                # 状态7：只依赖agent 2和4，agent 1和3用None
                a2 = np.random.randint(0, 4)
                a4 = np.random.randint(0, 4)
                opt_action = (None, a2, None, a4)

            else:
                # 其他状态：所有agent都有特定最优动作
                opt_action = tuple(np.random.randint(0, 4) for _ in range(num_agents))

            optimals.append(opt_action)

        return optimals

    def generate_def_task_optimals(self, main_optimals, chain_length=11):
        """
        生成防守任务(Chainball-11-Def)的optimals

        关键约束：状态1-4的最优动作必须与主任务的agent 1,2一致

        Args:
            main_optimals: 主任务的optimals

        Returns:
            list: 长度为chain_length+2的列表，每个元素是2个动作的元组
        """
        optimals = []

        for state in range(chain_length + 2):
            if 1 <= state <= 4:
                # 状态1-4：复制主任务中agent 1和2的最优动作
                # main_optimals[state] = (a1, a2, a3, a4) or with None
                main_opt = main_optimals[state]
                opt_action = (main_opt[0], main_opt[1])
            else:
                # 其他状态：随机生成2个agent的最优动作
                opt_action = tuple(np.random.randint(0, 4) for _ in range(2))

            optimals.append(opt_action)

        return optimals

    def generate_att_task_optimals(self, main_optimals, chain_length=11):
        """
        生成进攻任务(Chainball-11-Att)的optimals

        关键约束：状态8-11的最优动作必须与主任务的agent 3,4一致

        Args:
            main_optimals: 主任务的optimals

        Returns:
            list: 长度为chain_length+2的列表，每个元素是2个动作的元组
        """
        optimals = []

        for state in range(chain_length + 2):
            if 8 <= state <= 11:
                # 状态8-11：复制主任务中agent 3和4的最优动作
                main_opt = main_optimals[state]
                opt_action = (main_opt[2], main_opt[3])
            else:
                # 其他状态：随机生成2个agent的最优动作
                opt_action = tuple(np.random.randint(0, 4) for _ in range(2))

            optimals.append(opt_action)

        return optimals

    def generate_all(self, chain_length=11):
        """
        生成所有三个任务的optimals

        Returns:
            dict: 包含'main', 'def', 'att'三个键的字典
        """
        # 先生成主任务
        main_optimals = self.generate_main_task_optimals(
            num_agents=4,
            chain_length=chain_length
        )

        # 基于主任务生成防守和进攻任务
        def_optimals = self.generate_def_task_optimals(
            main_optimals,
            chain_length=chain_length
        )
        att_optimals = self.generate_att_task_optimals(
            main_optimals,
            chain_length=chain_length
        )

        return {
            'main': main_optimals,
            'def': def_optimals,
            'att': att_optimals
        }

    # def save_to_pickle(self, optimals_dict, filename='chainball_11_optimals.pkl'):
    #     """保存为pickle文件"""
    #     with open(filename, 'wb') as f:
    #         pickle.dump(optimals_dict, f)
    #     print(f"✓ Optimals saved to {filename}")

    def save_to_json(self, optimals_dict, filename='chainball_11_optimals.json'):
        """保存为JSON文件（便于查看）"""
        # 将None转换为字符串"None"以便JSON序列化
        json_dict = {}
        for task, optimals in optimals_dict.items():
            json_dict[task] = [
                [a if a is not None else "None" for a in opt]
                for opt in optimals
            ]

        with open(filename, 'w') as f:
            json.dump(json_dict, f, indent=2)
        print(f"✓ Optimals saved to {filename} (human-readable)")

    # def load_from_pickle(self, filename='chainball_11_optimals.pkl'):
    #     """从pickle文件加载"""
    #     with open(filename, 'rb') as f:
    #         optimals_dict = pickle.load(f)
    #     print(f"✓ Optimals loaded from {filename}")
    #     return optimals_dict

    def print_summary(self, optimals_dict):
        """打印optimals的摘要信息"""
        print("\n" + "=" * 60)
        print("Chainball-11 Optimals Summary")
        print("=" * 60)

        for task_name, optimals in optimals_dict.items():
            print(f"\n{task_name.upper()} Task:")
            print(f"  Total states: {len(optimals)}")
            print(f"  Agents per state: {len(optimals[0])}")

            # 显示前几个状态
            print(f"  First 5 states:")
            for i in range(min(5, len(optimals))):
                print(f"    State {i}: {optimals[i]}")

            # 检查关键状态
            if task_name == 'main':
                print(f"  Special states:")
                print(f"    State 5 (partial): {optimals[5]}")
                print(f"    State 7 (partial): {optimals[7]}")
            elif task_name == 'def':
                print(f"  Overlap states (1-4):")
                for i in range(1, 5):
                    print(f"    State {i}: {optimals[i]}")
            elif task_name == 'att':
                print(f"  Overlap states (8-11):")
                for i in range(8, 12):
                    print(f"    State {i}: {optimals[i]}")

        print("\n" + "=" * 60)


def verify_consistency(optimals_dict):
    """
    验证三个任务之间的一致性
    """
    main = optimals_dict['main']
    def_opt = optimals_dict['def']
    att_opt = optimals_dict['att']

    print("\nVerifying consistency...")

    # 检查防守任务状态1-4
    def_consistent = True
    for s in range(1, 5):
        if def_opt[s] != (main[s][0], main[s][1]):
            print(f"  ✗ State {s}: Def task inconsistent!")
            def_consistent = False
    if def_consistent:
        print("  ✓ Defense task states 1-4 consistent with main task")

    # 检查进攻任务状态8-11
    att_consistent = True
    for s in range(8, 12):
        if att_opt[s] != (main[s][2], main[s][3]):
            print(f"  ✗ State {s}: Att task inconsistent!")
            att_consistent = False
    if att_consistent:
        print("  ✓ Attack task states 8-11 consistent with main task")

    # 检查主任务的特殊状态
    if main[5][1] is None and main[5][3] is None:
        print("  ✓ Main task state 5 has partial dependency (a1, a3)")
    else:
        print("  ✗ Main task state 5 should have None for a2, a4")

    if main[7][0] is None and main[7][2] is None:
        print("  ✓ Main task state 7 has partial dependency (a2, a4)")
    else:
        print("  ✗ Main task state 7 should have None for a1, a3")

    return def_consistent and att_consistent


# ============ 使用示例 ============

if __name__ == "__main__":
    # 1. 生成optimals
    print("Generating Chainball-11 optimals...")
    generator = ChainballOptimalsGenerator(seed=42)
    optimals_dict = generator.generate_all(chain_length=11)

    # 2. 验证一致性
    verify_consistency(optimals_dict)

    # 3. 打印摘要
    generator.print_summary(optimals_dict)

    # 4. 保存文件
    # generator.save_to_pickle(optimals_dict, 'chainball_11_optimals.pkl')
    generator.save_to_json(optimals_dict, 'chainball_11_optimals.json')

    print("\n" + "=" * 60)
    print("Done! You can now use these optimals in your experiments:")
    print("=" * 60)
    print("""
# Load and use in your code:
import pickle

with open('chainball_11_optimals.pkl', 'rb') as f:
    optimals = pickle.load(f)

# Create environments
env_main = ChainballEnv(optimals=optimals['main'], ...)
env_def = ChainballEnv(optimals=optimals['def'], ...)
env_att = ChainballEnv(optimals=optimals['att'], ...)
    """)