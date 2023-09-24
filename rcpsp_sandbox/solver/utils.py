from instances.problem_instance import ResourceShiftMode


def resource_mode_operating_hours(mode: ResourceShiftMode) -> list[tuple]:
    if mode == ResourceShiftMode.SINGLE:
        return [(8, 16)]
    elif mode == ResourceShiftMode.DOUBLE:
        return [(6, 22)]
