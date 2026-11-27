"""Online clocks/keys; one advance per decision, independent environment resets."""


class MemoryState:
    def __init__(self, memory):
        self.memory = memory
        self.slots = {}  # (session, env) -> (episode key, next decision index)

    def context(self, batch_size, options=None):
        options = options or {}
        session = str(options.get("session_id", "standalone"))
        envs = options.get("env_ids", list(range(batch_size)))
        episodes = options.get("episode_ids", ["episode0"] * batch_size)
        if len(envs) != batch_size or len(episodes) != batch_size or len(set(envs)) != batch_size:
            raise ValueError("Each observation needs one distinct environment and Episode id")
        keys, times = [], []
        for env, episode in zip(envs, episodes):
            slot = (session, str(env))
            key = (*slot, str(episode))
            old = self.slots.get(slot)
            if old is not None and old[0] != key:
                self.memory.reset([old[0]])
                old = None
            self.slots[slot] = old or (key, 0)
            keys.append(key)
            times.append(self.slots[slot][1])
        return {"mode": "stream", "episode_keys": keys, "timesteps": times}

    def advance(self, context):
        for key, timestep in zip(context["episode_keys"], context["timesteps"]):
            self.slots[key[:2]] = (key, timestep + 1)

    def reset(self, options=None):
        if options is None:
            self.memory.reset()
            self.slots.clear()
            return
        session = str(options.get("session_id", "standalone"))
        envs = options.get("env_ids")
        wanted = None if envs is None else {str(env) for env in envs}
        slots = [slot for slot in self.slots
                 if slot[0] == session and (wanted is None or slot[1] in wanted)]
        self.memory.reset([self.slots[slot][0] for slot in slots])
        for slot in slots:
            del self.slots[slot]
