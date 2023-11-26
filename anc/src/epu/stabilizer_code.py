from typing import Iterable

import stim


class StabilizerCode:
    def __init__(
            self,
            *,
            obs_xs: Iterable[stim.PauliString],
            obs_zs: Iterable[stim.PauliString],
            stabilizers: Iterable[stim.PauliString],
    ):
        self.obs_xs = tuple(obs_xs)
        self.obs_zs = tuple(obs_zs)
        self.stabilizers = tuple(stabilizers)
        self.num_qubits = len(self.obs_xs[0])
        self.num_logical_qubits = len(self.obs_xs)

        # Verify this is actually a stabilizer code.
        assert len(self.obs_xs) == len(self.obs_zs)
        all_terms = [*self.obs_xs, *self.obs_zs, *self.stabilizers]
        for k1 in range(len(all_terms)):
            p1 = all_terms[k1]
            assert len(p1) == self.num_qubits
            for k2 in range(k1 + 1, len(all_terms)):
                is_obs_pair = k1 < len(self.obs_xs) and k2 == k1 + len(self.obs_xs)
                p2 = all_terms[k2]
                assert p1.commutes(p2) == (not is_obs_pair), (p1, p2)

    def detects_error(self, err: stim.PauliString) -> bool:
        assert len(err) == self.num_qubits
        return not all(s.commutes(err) for s in self.stabilizers)


rep_xx = StabilizerCode(
    obs_xs=[stim.PauliString('X_')],
    obs_zs=[stim.PauliString('ZY')],
    stabilizers=[stim.PauliString('XX')],
)
rep_yy = StabilizerCode(
    obs_xs=[stim.PauliString('XZ')],
    obs_zs=[stim.PauliString('ZZ')],
    stabilizers=[stim.PauliString('YY')],
)
rep_zz = StabilizerCode(
    obs_xs=[stim.PauliString('XY')],
    obs_zs=[stim.PauliString('Z_')],
    stabilizers=[stim.PauliString('ZZ')],
)
