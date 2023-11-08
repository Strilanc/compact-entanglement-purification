class ErrModel:
    """Digitized Bell pair noise model."""

    def __init__(
        self,
        *,
        x: float,
        y: float,
        z: float,
    ):
        """
        Args:
            x: Chance of the state |01> + |10>.
            y: Chance of the state |01> - |10>.
            z: Chance of the state |00> - |11>.
        """
        assert 0 <= x <= 1
        assert 0 <= y <= 1
        assert 0 <= z <= 1
        assert 0 <= x + y + z <= 1
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_odds(w: float, x: float, y: float, z: float) -> 'ErrModel':
        t = w + x + y + z
        return ErrModel(x=x / t, y=y / t, z=z / t)

    @property
    def w(self) -> float:
        return 1 - self.x - self.y - self.z

    def __str__(self):
        x = self.x
        y = self.y
        z = self.z
        return f'x={x} y={y} z={z}'
