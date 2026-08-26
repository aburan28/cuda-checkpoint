"""The address a process can be reached at, by whoever needs to know it."""

import os
import socket


def primary_ip(*overrides):
    """The source address the kernel would use for an outbound packet.

    Named environment variables win, in order, for hosts with several
    interfaces where the routing default is not the fabric the job uses -
    MNCR_NODE_IP for an agent, MNCR_RANK_IP for a rank. Otherwise a UDP
    socket "connects" to pick a route; no packet is sent.
    """
    for name in overrides:
        value = os.environ.get(name)
        if value:
            return value
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
