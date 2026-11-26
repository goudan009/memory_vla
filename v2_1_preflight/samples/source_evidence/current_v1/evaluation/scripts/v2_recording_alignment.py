"""Exact recording identity: one source file may contain several attempts."""
import numpy as np


def matching_demo(data, actions):
    matches = []
    inventory = []
    for name, group in data.items():
        if not name.startswith('demo_'):
            continue
        if 'action' not in group:
            inventory.append(dict(group=name,frames=None,complete=False))
            continue
        recorded = group['action']
        inventory.append(dict(group=name,frames=len(recorded),complete=True))
        if recorded.shape == actions.shape and np.array_equal(recorded[:],actions):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(f'Expected unique exact action recording, got {matches}; inventory={inventory}')
    group = data[matches[0]]
    if len(group['state']) != len(actions)+1 or len(group['state_size']) != len(actions)+1:
        raise ValueError('Matched recording lacks T+1 physical observation states')
    return matches[0],inventory
