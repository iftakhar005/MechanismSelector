"""mechanism_selector -- research artifact, not a production tool.

On a drift alarm, try a cheap incremental model update first, keep it if it
restores accuracy on held-out recent data, and rebuild from scratch only if it
does not. In the benchmarks this code was evaluated on, that policy did **not**
beat simpler baselines; see the package README before using it.

The public interface is `MechanismSelector` and its `adapt` method. The
mechanisms it needs are constructed with `mechanism_selector.mechanisms`.
"""

from .selector import MechanismSelector

__all__ = ["MechanismSelector"]
__version__ = "0.1.0"
