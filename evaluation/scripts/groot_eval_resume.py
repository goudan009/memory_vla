"""Plan a continuation from recorded slot prefixes without replaying completed episodes."""
import copy


def continuation(result, records, additional=None):
    config = result['configuration']
    n_envs = int(config['n_envs_per_gpu'])
    instances = {str(gpu): list(ids) for gpu, ids in config['instances'].items()}
    layout = copy.deepcopy(config.get('episode_layout') or {})
    for gpu, ids in instances.items():
        if gpu not in layout:
            width = len(ids) // n_envs
            layout[gpu] = [dict(instances=ids[i*width:(i+1)*width] if i < n_envs-1 else ids[i*width:],
                                quota=0, offset=0) for i in range(n_envs)]
            per_gpu = int(result['requested_episodes']) // len(instances)
            for i, slot in enumerate(layout[gpu]):
                slot['quota'] = per_gpu//n_envs + (per_gpu % n_envs if i == n_envs-1 else 0)
    counts = {(gpu,i): 0 for gpu in instances for i in range(n_envs)}
    for record in records:
        gpu = str(record.get('gpu_index', next(iter(instances))))
        index = record.get('env_idx')
        if (gpu,index) not in counts:
            raise ValueError('回合记录与原 GPU / 环境分配不一致')
        if not record.get('valid', False):
            raise ValueError('存在无效回合，请先检查原因后再续跑')
        count = counts[gpu,index]
        slot = layout[gpu][index]
        if count >= slot['quota'] or int(record['instance_id']) != slot['instances'][count % len(slot['instances'])]:
            raise ValueError('回合记录不是原环境中已完成的连续前缀')
        counts[gpu,index] += 1
    if additional is not None:
        if type(additional) is not int or additional < 1:
            raise ValueError('本次回合数必须为正整数')
        # Balance the added work; favor slots with fewer completed episodes.
        ordered = sorted(counts, key=lambda key: (counts[key], key))
        each, extra = divmod(additional, len(ordered))
        budgets = {key: each + (i < extra) for i,key in enumerate(ordered)}
        for (gpu,index), count in counts.items():
            layout[gpu][index]['quota'] = count + budgets[gpu,index]
    else:
        budgets = {key: layout[key[0]][key[1]]['quota']-count for key,count in counts.items()}
    children = {}
    for gpu in instances:
        active = []
        for index, slot in enumerate(layout[gpu]):
            slot['offset'] = 0
            if budgets[gpu,index]:
                active.append(dict(slot, offset=counts[gpu,index], quota=budgets[gpu,index], env_index=index))
        if active:
            children[gpu] = active
    if not children:
        raise ValueError('原计划已完成，请填写本次再跑的回合数')
    return dict(episode_layout=layout, child_layout=children,
                recorded_episodes=len(records), additional_episodes=sum(budgets.values()),
                requested_episodes=len(records)+sum(budgets.values()))
