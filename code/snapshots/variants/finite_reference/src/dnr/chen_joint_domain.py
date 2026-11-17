"""Common graph domain: original NC34 legal moves intersected with Chen (47)."""
from .nature34_clpu_environment import CLPUEnvironment


class ChenEnv(CLPUEnvironment):
    @classmethod
    def from_env(cls, env):
        return cls(env.graph, env.static, env.loads, env.scenario, env.condition, env.event,
                   horizon=env.horizon, slot_seconds=env.slot_seconds, sample_seconds=env.sample_seconds)

    def live_mask(self, command):
        """Energized controlled lines, irrespective of individual load readiness."""
        switch_mask = command >> 10
        if not hasattr(self, "_live_cache"):
            self._live_cache = {}
        if switch_mask not in self._live_cache:
            buses = set(self.metadata(command)["source_reachable_buses"])
            live = sum(1 << i for i, (u, v) in enumerate(self.static["switch_edges"])
                       if switch_mask & (1 << i) and u in buses and v in buses)
            self._live_cache[switch_mask] = live
        return self._live_cache[switch_mask]

    def legal_actions(self, state):
        before = self.live_mask(state.command)
        return [a for a in super().legal_actions(state) if a <= 10 or
                not before & ~self.live_mask(state.command ^ (1 << (a - 1)))]
