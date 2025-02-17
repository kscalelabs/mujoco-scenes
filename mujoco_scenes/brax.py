"""Utilities for converting between MuJoCo and Brax physics engines.

This module provides functions to:
1. Convert MuJoCo models to Brax System objects
2. Handle custom parameters and scaling factors
3. Process joints, actuators and other physical properties
4. Validate model compatibility between the two engines

The main entry point is the load_model() function.
"""

import itertools

import mujoco
import numpy as np
from mujoco import mjx

try:
    import jax
    from brax.base import Actuator, DoF, Inertia, Link, Motion, System, Transform
    from jax import numpy as jnp
except ImportError as e:
    raise ImportError("You must install `brax` and `jax` to use this module.") from e


def _get_name(mj: mujoco.MjModel, i: int) -> str:
    """Extracts a null-terminated string from MuJoCo's names buffer.

    Args:
        mj: MuJoCo model instance
        i: Starting index in the names buffer

    Returns:
        String from the buffer up to the first null character
    """
    names = mj.names[i:].decode("utf-8")
    return names[: names.find("\x00")]


def _check_custom(mj: mujoco.MjModel, custom: dict[str, np.ndarray]) -> None:
    """Validates custom parameters for Brax conversion.

    Checks:
    - Spring mass and inertia scales are in [0,1]
    - Initial joint positions match model DOFs

    Args:
        mj: MuJoCo model instance
        custom: Dictionary of custom parameters

    Raises:
        ValueError: If parameters are invalid
    """
    if not (0 <= custom["spring_mass_scale"] <= 1 and 0 <= custom["spring_inertia_scale"] <= 1):
        raise ValueError("Spring inertia and mass scale must be in [0, 1].")
    if "init_qpos" in custom and custom["init_qpos"].shape[0] != mj.nq:
        size = custom["init_qpos"].shape[0]
        raise ValueError(f"init_qpos had length {size} but expected length {mj.nq}.")


def _get_custom(mj: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Extracts custom parameters from MuJoCo model for Brax conversion.

    Processes:
    - Global parameters (damping, ERP, scales)
    - Per-body parameters (constraints, stiffness)
    - Per-geom parameters (elasticity)

    Default values are provided for missing parameters.

    Args:
        mj: MuJoCo model instance

    Returns:
        Dictionary of processed custom parameters
    """
    default = {
        "ang_damping": (0.0, None),
        "vel_damping": (0.0, None),
        "baumgarte_erp": (0.1, None),
        "spring_mass_scale": (0.0, None),
        "spring_inertia_scale": (0.0, None),
        "joint_scale_pos": (0.5, None),
        "joint_scale_ang": (0.2, None),
        "collide_scale": (1.0, None),
        "matrix_inv_iterations": (10, None),
        "solver_maxls": (20, None),
        "elasticity": (0.0, "geom"),
        "constraint_stiffness": (2000.0, "body"),
        "constraint_limit_stiffness": (1000.0, "body"),
        "constraint_ang_damping": (0.0, "body"),
        "constraint_vel_damping": (0.0, "body"),
    }

    # add user provided overrides to the defaults
    for i, ni in enumerate(mj.name_numericadr):
        nsize = mj.numeric_size[i]
        name = _get_name(mj, ni)
        val = mj.numeric_data[mj.numeric_adr[i] : mj.numeric_adr[i] + nsize]
        typ = default[name][1] if name in default else None
        default[name] = (val, typ)

    # gather custom overrides with correct sizes
    custom = {}
    for name, (val, typ) in default.items():
        val = np.array([val])
        size = {
            "body": mj.nbody - 1,  # ignore the world body
            "geom": mj.ngeom,
        }.get(typ or "", val.shape[-1])
        if val.shape[-1] != size and val.shape[-1] > 1:
            # the provided shape does not match against our default size
            raise ValueError(
                f'"{name}" custom arg needed {size} values for the "{typ}" type, but got {val.shape[-1]} values.'
            )
        elif val.shape[-1] != size and val.shape[-1] == 1:
            val = np.repeat(val, size)
        val = val.squeeze() if not typ else val.reshape(size)
        if typ == "body":
            # pad one value for the world body, which gets dropped at Link creation
            val = np.concatenate([[val[0]], val])
        custom[name] = val

    # get tuple custom overrides
    for i, ni in enumerate(mj.name_tupleadr):
        start, end = mj.tuple_adr[i], mj.tuple_adr[i] + mj.tuple_size[i]
        objtype = mj.tuple_objtype[start:end]
        name = _get_name(mj, ni)
        if not all(objtype[0] == objtype):
            raise NotImplementedError(f'All tuple elements "{name}" should have the same object type.')
        if objtype[0] not in [1, 5]:
            raise NotImplementedError(f'Custom tuple "{name}" with objtype=={objtype[0]} is not supported.')
        typ = {1: "body", 5: "geom"}[objtype[0]]
        if name in default and default[name][1] != typ:
            raise ValueError(f'Custom tuple "{name}" is expected to be associated with the {default[name][1]} objtype.')

        size = {1: mj.nbody, 5: mj.ngeom}[objtype[0]]
        default_val, _ = default.get(name, (0.0, None))
        arr = np.repeat(default_val, size)
        objid = mj.tuple_objid[start:end]
        objprm = mj.tuple_objprm[start:end]
        arr[objid] = objprm
        custom[name] = arr

    _check_custom(mj, custom)
    return custom


def load_model(mj: mujoco.MjModel) -> System:
    """Creates a brax system from a MuJoCo model."""
    custom = _get_custom(mj)

    # Create links
    joint_positions = [np.array([0.0, 0.0, 0.0])]
    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]):
        position = np.array([p for _, p in group])
        joint_positions.append(position[0])
    joint_position = np.array(joint_positions)
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (mj.nbody, 1))
    link = Link(  # pytype: disable=wrong-arg-types  # jax-ndarray
        transform=Transform(pos=mj.body_pos, rot=mj.body_quat),  # pytype: disable=wrong-arg-types  # jax-ndarray
        inertia=Inertia(  # pytype: disable=wrong-arg-types  # jax-ndarray
            transform=Transform(pos=mj.body_ipos, rot=mj.body_iquat),  # pytype: disable=wrong-arg-types  # jax-ndarray
            i=np.array([np.diag(i) for i in mj.body_inertia]),
            mass=mj.body_mass,
        ),
        invweight=mj.body_invweight0[:, 0],
        joint=Transform(pos=joint_position, rot=identity),  # pytype: disable=wrong-arg-types  # jax-ndarray
        constraint_stiffness=custom["constraint_stiffness"],
        constraint_vel_damping=custom["constraint_vel_damping"],
        constraint_limit_stiffness=custom["constraint_limit_stiffness"],
        constraint_ang_damping=custom["constraint_ang_damping"],
    )

    # Skip link 0 which is the world body in MuJoCo.
    # Copy to avoid writing to MuJoCo model.
    link = jax.tree.map(lambda x: x[1:].copy(), link)

    # Create DOFs.
    mj.jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
    motions, limits, stiffnesses = [], [], []
    for typ, axis, limit, stiffness in zip(mj.jnt_type, mj.jnt_axis, mj.jnt_range, mj.jnt_stiffness):
        if typ == 0:
            motion = Motion(ang=np.eye(6, 3, -3), vel=np.eye(6, 3))
            limit = np.array([-np.inf] * 6), np.array([np.inf] * 6)
            stiffness = np.zeros(6)
        elif typ == 1:
            motion = Motion(ang=np.eye(3), vel=np.zeros((3, 3)))
            limit = np.array([-np.inf] * 3), np.array([np.inf] * 3)
            stiffness = np.zeros(3)
        elif typ == 2:
            motion = Motion(ang=np.zeros((1, 3)), vel=axis.reshape((1, 3)))
            limit = limit[0:1], limit[1:2]
            stiffness = np.array([stiffness])
        elif typ == 3:
            motion = Motion(ang=axis.reshape((1, 3)), vel=np.zeros((1, 3)))
            limit = limit[0:1], limit[1:2]
            stiffness = np.array([stiffness])
        else:
            # Invalid joint type
            continue
        motions.append(motion)
        limits.append(limit)
        stiffnesses.append(stiffness)
    motion = jax.tree.map(lambda *x: np.concatenate(x), *motions)

    limit = None
    if np.any(mj.jnt_limited):
        limit = jax.tree.map(lambda *x: np.concatenate(x), *limits)
    stiffness = np.concatenate(stiffnesses)
    solver_params_jnt = np.concatenate((mj.jnt_solref, mj.jnt_solimp), axis=1)
    solver_params_dof = solver_params_jnt[mj.dof_jntid]

    dof = DoF(  # pytype: disable=wrong-arg-types
        motion=motion,
        armature=mj.dof_armature,
        stiffness=stiffness,
        damping=mj.dof_damping,
        limit=limit,
        invweight=mj.dof_invweight0,
        solver_params=solver_params_dof,
    )

    # Create actuators.
    ctrl_range = mj.actuator_ctrlrange
    ctrl_range[~(mj.actuator_ctrllimited == 1), :] = np.array([-np.inf, np.inf])
    force_range = mj.actuator_forcerange
    force_range[~(mj.actuator_forcelimited == 1), :] = np.array([-np.inf, np.inf])
    bias_q = mj.actuator_biasprm[:, 1] * (mj.actuator_biastype != 0)
    bias_qd = mj.actuator_biasprm[:, 2] * (mj.actuator_biastype != 0)

    # Mask actuators since Brax only supports joint transmission types.
    act_mask = mj.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
    trnid = mj.actuator_trnid[act_mask, 0].astype(np.uint32)
    q_id = mj.jnt_qposadr[trnid]
    qd_id = mj.jnt_dofadr[trnid]
    act_kwargs = {
        "gain": mj.actuator_gainprm[:, 0],
        "gear": mj.actuator_gear[:, 0],
        "ctrl_range": ctrl_range,
        "force_range": force_range,
        "bias_q": bias_q,
        "bias_qd": bias_qd,
    }
    act_kwargs = jax.tree.map(lambda x: x[act_mask], act_kwargs)

    actuator = Actuator(q_id=q_id, qd_id=qd_id, **act_kwargs)  # pytype: disable=wrong-arg-types

    # Create non-pytree params. These do not live on device directly, and they
    # cannot be differentiated, but they do change the emitted control flow.
    link_names = [_get_name(mj, i) for i in mj.name_bodyadr[1:]]

    # Convert stacked joints to 1, 2, or 3
    link_types = ""
    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]):
        typs = [t for _, t in group]
        if len(typs) == 1 and typs[0] == 0:  # free
            typ = "f"
        elif 0 in typs or 1 in typs:
            # Invalid joint stack
            continue
        else:
            typ = str(len(typs))
        link_types += typ
    link_parents = tuple(mj.body_parentid - 1)[1:]

    mjx_model = mjx.put_model(mj)

    sys = System(  # pytype: disable=wrong-arg-types  # jax-ndarray
        gravity=mj.opt.gravity,
        viscosity=mj.opt.viscosity,
        density=mj.opt.density,
        elasticity=custom["elasticity"],
        link=link,
        dof=dof,
        actuator=actuator,
        init_q=custom["init_qpos"] if "init_qpos" in custom else mj.qpos0,
        vel_damping=custom["vel_damping"],
        ang_damping=custom["ang_damping"],
        baumgarte_erp=custom["baumgarte_erp"],
        spring_mass_scale=custom["spring_mass_scale"],
        spring_inertia_scale=custom["spring_inertia_scale"],
        joint_scale_ang=custom["joint_scale_ang"],
        joint_scale_pos=custom["joint_scale_pos"],
        collide_scale=custom["collide_scale"],
        enable_fluid=(mj.opt.viscosity > 0) | (mj.opt.density > 0),
        link_names=link_names,
        link_types=link_types,
        link_parents=link_parents,
        matrix_inv_iterations=int(custom["matrix_inv_iterations"]),
        solver_iterations=mj.opt.iterations,
        solver_maxls=int(custom["solver_maxls"]),
        mj_model=mj,
        **mjx_model.__dict__,
    )

    sys = jax.tree.map(jnp.array, sys)

    return sys
