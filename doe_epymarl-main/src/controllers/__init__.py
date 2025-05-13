REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC

from .doe_controller import DoEMAC
from .non_shared_doe_controller import DoENonSharedMAC
REGISTRY["doe_mac"] = DoEMAC
REGISTRY["non_shared_doe_mac"] = DoENonSharedMAC