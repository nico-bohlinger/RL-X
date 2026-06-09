"""External monkey-patch for MuJoCo Warp hfield collision to support per-world hfields.

Design
------
The MuJoCo Warp collision kernel for hfields uses ``geom_dataid[g1]`` to select the
hfield for ALL worlds — there is no per-world routing.  To support per-world hfields
(one hfield copy per env) we use the following layout:

  * Build the MuJoCo model with ``nr_envs * num_arms`` hfields (named
    ``{base}_tool_hfield_0`` … ``{base}_tool_hfield_{nr_envs-1}`` per arm).
  * Arm *a*'s hfields for envs 0…N-1 have sequential IDs starting at
    ``geom_dataid[arm_a_geom]`` (i.e. the default 1D geom_dataid already points to
    hfield 0 for each arm).
  * The patched kernel does ``geom_dataid[g1] + worldid`` so world *i* selects
    hfield ``base_id + i``, which is that world's hfield.

Call ``apply_patch()`` **once**, before the first ``mjx.step()`` invocation.
"""

import functools

import warp as wp
from typing import Tuple

import mujoco.mjx.third_party.mujoco_warp._src.collision_convex as _cc
import mujoco.mjx.third_party.mujoco_warp._src.warp_util as _wu

_patched = False


def apply_patch(hfield_nrow: int = 16, hfield_ncol: int = 16, mj_maxconpair: int = None):
    global _patched
    if _patched:
        return
    _patched = True

    # --- pull all symbols used inside the two functions we're rewriting ----------
    MJ_MAXVAL = _cc.MJ_MAXVAL
    # MJ_MAXCONPAIR sizes a fixed per-thread local array holding the candidate contacts of a single
    # geom/hfield pair. The kernel only walks the colliding geom's bounding-box cells, so it must be
    # bounded by the geom footprint (number of hfield cells under one foot), NOT the full grid -
    # nrow*ncol would be a huge per-thread array at locomotion terrain resolutions. The caller passes
    # an explicit value computed from the foot size and the hfield cell size.
    if mj_maxconpair is None:
        mj_maxconpair = max(64, hfield_nrow * hfield_ncol)
    MJ_MAXCONPAIR = mj_maxconpair
    GeomType = _cc.GeomType
    Geom = _cc.Geom
    vec5 = _cc.vec5
    mat63 = _cc.mat63
    vec_maxconpair = wp.types.vector(MJ_MAXCONPAIR, float)
    mat_maxconpair = wp.types.matrix((MJ_MAXCONPAIR, 3), float)
    support = _cc.support
    ccd = _cc.ccd
    make_frame = _cc.make_frame
    write_contact = _cc.write_contact
    contact_params = _cc.contact_params
    geom_collision_pair = _cc.geom_collision_pair
    cache_kernel = _wu.cache_kernel

    # --------------------------------------------------------------------------
    # Patched _hfield_filter: uses geom_dataid[g1] + worldid
    # --------------------------------------------------------------------------
    @wp.func
    def _hfield_filter_perenv(
        geom_type: wp.array(dtype=int),
        geom_dataid: wp.array(dtype=int),
        geom_size: wp.array2d(dtype=wp.vec3),
        geom_rbound: wp.array2d(dtype=float),
        geom_margin: wp.array2d(dtype=float),
        mesh_vertadr: wp.array(dtype=int),
        mesh_vertnum: wp.array(dtype=int),
        mesh_graphadr: wp.array(dtype=int),
        mesh_vert: wp.array(dtype=wp.vec3),
        mesh_graph: wp.array(dtype=int),
        hfield_size: wp.array(dtype=wp.vec4),
        geom_xpos_in: wp.array2d(dtype=wp.vec3),
        geom_xmat_in: wp.array2d(dtype=wp.mat33),
        worldid: int,
        g1: int,
        g2: int,
    ) -> Tuple[bool, float, float, float, float, float, float]:
        hfdataid = geom_dataid[g1] + worldid  # PER-WORLD: offset by worldid
        size1 = hfield_size[hfdataid]

        rbound_id = worldid % geom_rbound.shape[0]
        margin_id = worldid % geom_margin.shape[0]
        size_id = worldid % geom_size.shape[0]

        pos1 = geom_xpos_in[worldid, g1]
        mat1 = geom_xmat_in[worldid, g1]
        mat1T = wp.transpose(mat1)
        pos2 = geom_xpos_in[worldid, g2]
        pos = mat1T @ (pos2 - pos1)
        r2 = geom_rbound[rbound_id, g2]

        margin = geom_margin[margin_id, g1] + geom_margin[margin_id, g2]

        for i in range(2):
            if (size1[i] < pos[i] - r2 - margin) or (-size1[i] > pos[i] + r2 + margin):
                return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

        if size1[2] < pos[2] - r2 - margin:
            return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

        if -size1[3] > pos[2] + r2 + margin:
            return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL

        mat2 = geom_xmat_in[worldid, g2]
        mat = mat1T @ mat2

        geom2 = Geom()
        geom2.pos = pos
        geom2.rot = mat
        geom2.size = geom_size[size_id, g2]
        geom2.margin = 0.0
        geom2.index = -1

        geomtype2 = geom_type[g2]

        if geomtype2 == GeomType.MESH:
            dataid = geom_dataid[g2]
            geom2.vertadr = wp.where(dataid >= 0, mesh_vertadr[dataid], -1)
            geom2.vertnum = wp.where(dataid >= 0, mesh_vertnum[dataid], -1)
            geom2.graphadr = wp.where(dataid >= 0, mesh_graphadr[dataid], -1)
            geom2.vert = mesh_vert
            geom2.graph = mesh_graph

        xmax = support(geom2, geomtype2, wp.vec3(1.0, 0.0, 0.0)).point[0]
        xmin = support(geom2, geomtype2, wp.vec3(-1.0, 0.0, 0.0)).point[0]
        ymax = support(geom2, geomtype2, wp.vec3(0.0, 1.0, 0.0)).point[1]
        ymin = support(geom2, geomtype2, wp.vec3(0.0, -1.0, 0.0)).point[1]
        zmax = support(geom2, geomtype2, wp.vec3(0.0, 0.0, 1.0)).point[2]
        zmin = support(geom2, geomtype2, wp.vec3(0.0, 0.0, -1.0)).point[2]

        if (
            (xmin - margin > size1[0])
            or (xmax + margin < -size1[0])
            or (ymin - margin > size1[1])
            or (ymax + margin < -size1[1])
            or (zmin - margin > size1[2])
            or (zmax + margin < -size1[3])
        ):
            return True, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL
        else:
            return False, xmin, xmax, ymin, ymax, zmin, zmax

    # --------------------------------------------------------------------------
    # Patched ccd_hfield_kernel_builder: uses geom_dataid[g1] + worldid
    # --------------------------------------------------------------------------
    @cache_kernel
    def ccd_hfield_kernel_builder_perenv(
        geomtype1: int,
        geomtype2: int,
        gjk_iterations: int,
        epa_iterations: int,
        geomgeomid: int,
    ):
        @wp.kernel(module="unique", enable_backward=False)
        def ccd_hfield_kernel(
            opt_ccd_tolerance: wp.array(dtype=float),
            geom_type: wp.array(dtype=int),
            geom_condim: wp.array(dtype=int),
            geom_dataid: wp.array(dtype=int),
            geom_priority: wp.array(dtype=int),
            geom_solmix: wp.array2d(dtype=float),
            geom_solref: wp.array2d(dtype=wp.vec2),
            geom_solimp: wp.array2d(dtype=vec5),
            geom_size: wp.array2d(dtype=wp.vec3),
            geom_rbound: wp.array2d(dtype=float),
            geom_friction: wp.array2d(dtype=wp.vec3),
            geom_margin: wp.array2d(dtype=float),
            geom_gap: wp.array2d(dtype=float),
            mesh_vertadr: wp.array(dtype=int),
            mesh_vertnum: wp.array(dtype=int),
            mesh_graphadr: wp.array(dtype=int),
            mesh_vert: wp.array(dtype=wp.vec3),
            mesh_graph: wp.array(dtype=int),
            mesh_polynum: wp.array(dtype=int),
            mesh_polyadr: wp.array(dtype=int),
            mesh_polynormal: wp.array(dtype=wp.vec3),
            mesh_polyvertadr: wp.array(dtype=int),
            mesh_polyvertnum: wp.array(dtype=int),
            mesh_polyvert: wp.array(dtype=int),
            mesh_polymapadr: wp.array(dtype=int),
            mesh_polymapnum: wp.array(dtype=int),
            mesh_polymap: wp.array(dtype=int),
            hfield_size: wp.array(dtype=wp.vec4),
            hfield_nrow: wp.array(dtype=int),
            hfield_ncol: wp.array(dtype=int),
            hfield_adr: wp.array(dtype=int),
            hfield_data: wp.array(dtype=float),
            pair_dim: wp.array(dtype=int),
            pair_solref: wp.array2d(dtype=wp.vec2),
            pair_solreffriction: wp.array2d(dtype=wp.vec2),
            pair_solimp: wp.array2d(dtype=vec5),
            pair_margin: wp.array2d(dtype=float),
            pair_gap: wp.array2d(dtype=float),
            pair_friction: wp.array2d(dtype=vec5),
            geom_xpos_in: wp.array2d(dtype=wp.vec3),
            geom_xmat_in: wp.array2d(dtype=wp.mat33),
            naconmax_in: int,
            naccdmax_in: int,
            ncollision_in: wp.array(dtype=int),
            collision_pair_in: wp.array(dtype=wp.vec2i),
            collision_pairid_in: wp.array(dtype=wp.vec2i),
            collision_worldid_in: wp.array(dtype=int),
            epa_vert_in: wp.array2d(dtype=wp.vec3),
            epa_vert_index_in: wp.array2d(dtype=int),
            epa_face_in: wp.array2d(dtype=int),
            epa_pr_in: wp.array2d(dtype=wp.vec3),
            epa_norm2_in: wp.array2d(dtype=float),
            epa_horizon_in: wp.array2d(dtype=int),
            nccd_in: wp.array(dtype=int),
            contact_dist_out: wp.array(dtype=float),
            contact_pos_out: wp.array(dtype=wp.vec3),
            contact_frame_out: wp.array(dtype=wp.mat33),
            contact_includemargin_out: wp.array(dtype=float),
            contact_friction_out: wp.array(dtype=vec5),
            contact_solref_out: wp.array(dtype=wp.vec2),
            contact_solreffriction_out: wp.array(dtype=wp.vec2),
            contact_solimp_out: wp.array(dtype=vec5),
            contact_dim_out: wp.array(dtype=int),
            contact_geom_out: wp.array(dtype=wp.vec2i),
            contact_efc_address_out: wp.array2d(dtype=int),
            contact_worldid_out: wp.array(dtype=int),
            contact_type_out: wp.array(dtype=int),
            contact_geomcollisionid_out: wp.array(dtype=int),
            nacon_out: wp.array(dtype=int),
        ):
            collisionid = wp.tid()
            if collisionid >= ncollision_in[0]:
                return

            geoms = collision_pair_in[collisionid]
            g1 = geoms[0]
            g2 = geoms[1]

            if geom_type[g1] != geomtype1 or geom_type[g2] != geomtype2:
                return

            worldid = collision_worldid_in[collisionid]

            no_hf_collision, xmin, xmax, ymin, ymax, zmin, zmax = _hfield_filter_perenv(
                geom_type,
                geom_dataid,
                geom_size,
                geom_rbound,
                geom_margin,
                mesh_vertadr,
                mesh_vertnum,
                mesh_graphadr,
                mesh_vert,
                mesh_graph,
                hfield_size,
                geom_xpos_in,
                geom_xmat_in,
                worldid,
                g1,
                g2,
            )
            if no_hf_collision:
                return

            ccdid = wp.atomic_add(nccd_in, wp.static(geomgeomid), 1)
            if ccdid >= naccdmax_in:
                wp.printf("CCD overflow - please increase naccdmax to %u\n", ccdid)
                return

            _, margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
                geom_condim,
                geom_priority,
                geom_solmix,
                geom_solref,
                geom_solimp,
                geom_friction,
                geom_margin,
                geom_gap,
                pair_dim,
                pair_solref,
                pair_solreffriction,
                pair_solimp,
                pair_margin,
                pair_gap,
                pair_friction,
                collision_pair_in,
                collision_pairid_in,
                collisionid,
                worldid,
            )

            geom1, geom2 = geom_collision_pair(
                geom_type,
                geom_dataid,
                geom_size,
                mesh_vertadr,
                mesh_vertnum,
                mesh_graphadr,
                mesh_vert,
                mesh_graph,
                mesh_polynum,
                mesh_polyadr,
                mesh_polynormal,
                mesh_polyvertadr,
                mesh_polyvertnum,
                mesh_polyvert,
                mesh_polymapadr,
                mesh_polymapnum,
                mesh_polymap,
                geom_xpos_in,
                geom_xmat_in,
                geoms,
                worldid,
            )

            hf_pos = geom_xpos_in[worldid, g1]
            hf_mat = geom_xmat_in[worldid, g1]
            hf_matT = wp.transpose(hf_mat)

            geom2.pos = hf_matT @ (geom2.pos - hf_pos)
            geom2.rot = hf_matT @ geom2.rot

            geom1.pos = wp.vec3(0.0, 0.0, 0.0)
            geom1.rot = wp.identity(n=3, dtype=float)

            geom1_dataid = geom_dataid[g1] + worldid  # PER-WORLD: offset by worldid

            nrow = hfield_nrow[geom1_dataid]
            ncol = hfield_ncol[geom1_dataid]
            size = hfield_size[geom1_dataid]

            x_scale = 0.5 * float(ncol - 1) / size[0]
            y_scale = 0.5 * float(nrow - 1) / size[1]
            cmin = wp.max(0, int(wp.floor((xmin + size[0]) * x_scale)))
            cmax = wp.min(ncol - 1, int(wp.ceil((xmax + size[0]) * x_scale)))
            rmin = wp.max(0, int(wp.floor((ymin + size[1]) * y_scale)))
            rmax = wp.min(nrow - 1, int(wp.ceil((ymax + size[1]) * y_scale)))

            dx = (2.0 * size[0]) / float(ncol - 1)
            dy = (2.0 * size[1]) / float(nrow - 1)
            dr = wp.vec2i(1, 0)

            prism = mat63()

            prism[0, 2] = -size[3]
            prism[1, 2] = -size[3]
            prism[2, 2] = -size[3]

            adr = hfield_adr[geom1_dataid]

            hfield_contact_dist = vec_maxconpair()
            hfield_contact_pos = mat_maxconpair()
            hfield_contact_normal = mat_maxconpair()
            min_dist = float(MJ_MAXVAL)
            min_normal = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
            min_pos = wp.vec3(MJ_MAXVAL, MJ_MAXVAL, MJ_MAXVAL)
            min_id = int(-1)

            geom2.margin = margin

            epa_vert = epa_vert_in[ccdid]
            epa_vert_index = epa_vert_index_in[ccdid]
            epa_face = epa_face_in[ccdid]
            epa_pr = epa_pr_in[ccdid]
            epa_norm2 = epa_norm2_in[ccdid]
            epa_horizon = epa_horizon_in[ccdid]

            collision_pairid = collision_pairid_in[collisionid]

            count = int(0)
            for r in range(rmin, rmax):
                for init_i in range(2):
                    x = dx * float(cmin) - size[0]
                    y = dy * float(r + dr[init_i]) - size[1]
                    z = hfield_data[adr + (r + dr[init_i]) * ncol + cmin] * size[2] + margin

                    prism[0] = prism[1]
                    prism[1] = prism[2]
                    prism[3] = prism[4]
                    prism[4] = prism[5]

                    prism[2, 0] = x
                    prism[5, 0] = x
                    prism[2, 1] = y
                    prism[5, 1] = y
                    prism[5, 2] = z

                for c in range(cmin + 1, cmax + 1):
                    for i in range(2):
                        if count >= MJ_MAXCONPAIR:
                            wp.printf(
                                "height field collision overflow, number of collisions >= %u - please adjust resolution: \n decrease the number of hfield rows/cols or modify size of colliding geom\n",
                                MJ_MAXCONPAIR,
                            )
                            continue

                        x = dx * float(c) - size[0]
                        y = dy * float(r + dr[i]) - size[1]
                        z = hfield_data[adr + (r + dr[i]) * ncol + c] * size[2] + margin

                        prism[0] = prism[1]
                        prism[1] = prism[2]
                        prism[3] = prism[4]
                        prism[4] = prism[5]

                        prism[2, 0] = x
                        prism[5, 0] = x
                        prism[2, 1] = y
                        prism[5, 1] = y
                        prism[5, 2] = z

                        if prism[3, 2] < zmin and prism[4, 2] < zmin and prism[5, 2] < zmin:
                            continue

                        geom1.hfprism = prism

                        x1 = geom1.pos + geom1.rot @ (prism[0] + prism[1] + prism[2] + prism[3] + prism[4] + prism[5]) * wp.static(1.0 / 6.0)

                        dist, ncontact, w1, w2, idx = ccd(
                            opt_ccd_tolerance[worldid % opt_ccd_tolerance.shape[0]],
                            0.0,
                            gjk_iterations,
                            epa_iterations,
                            geom1,
                            geom2,
                            geomtype1,
                            geomtype2,
                            x1,
                            geom2.pos,
                            epa_vert,
                            epa_vert_index,
                            epa_face,
                            epa_pr,
                            epa_norm2,
                            epa_horizon,
                        )

                        if ncontact == 0:
                            continue

                        hfield_contact_dist[count] = dist

                        pos_local = 0.5 * (w1 + w2)
                        pos = hf_mat @ pos_local + hf_pos
                        hfield_contact_pos[count, 0] = pos[0]
                        hfield_contact_pos[count, 1] = pos[1]
                        hfield_contact_pos[count, 2] = pos[2]

                        frame_local = make_frame(w1 - w2)
                        normal_local = wp.vec3(frame_local[0, 0], frame_local[0, 1], frame_local[0, 2])
                        normal = hf_mat @ normal_local
                        hfield_contact_normal[count, 0] = normal[0]
                        hfield_contact_normal[count, 1] = normal[1]
                        hfield_contact_normal[count, 2] = normal[2]

                        if dist < min_dist:
                            min_dist = dist
                            min_normal = normal
                            min_pos = pos
                            min_id = count

                        count += 1

            write_contact(
                naconmax_in,
                0,
                min_dist,
                min_pos,
                make_frame(min_normal),
                margin,
                gap,
                condim,
                friction,
                solref,
                solreffriction,
                solimp,
                geoms,
                collision_pairid,
                worldid,
                contact_dist_out,
                contact_pos_out,
                contact_frame_out,
                contact_includemargin_out,
                contact_friction_out,
                contact_solref_out,
                contact_solreffriction_out,
                contact_solimp_out,
                contact_dim_out,
                contact_geom_out,
                contact_efc_address_out,
                contact_worldid_out,
                contact_type_out,
                contact_geomcollisionid_out,
                nacon_out,
            )

            if wp.static(True):
                MIN_DIST_TO_NEXT_CONTACT = 1.0e-3

                id1 = int(-1)
                dist1 = float(-MJ_MAXVAL)
                for i in range(count):
                    if i == min_id:
                        continue
                    hf_pos_i = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
                    dist = wp.norm_l2(hf_pos_i - min_pos)
                    if dist > dist1:
                        id1 = i
                        dist1 = dist

                if id1 == -1 or (0.0 < dist1 and dist1 < MIN_DIST_TO_NEXT_CONTACT):
                    return

                pos1 = wp.vec3(hfield_contact_pos[id1, 0], hfield_contact_pos[id1, 1], hfield_contact_pos[id1, 2])
                normal1 = wp.vec3(hfield_contact_normal[id1, 0], hfield_contact_normal[id1, 1], hfield_contact_normal[id1, 2])

                write_contact(
                    naconmax_in,
                    1,
                    hfield_contact_dist[id1],
                    pos1,
                    make_frame(normal1),
                    margin,
                    gap,
                    condim,
                    friction,
                    solref,
                    solreffriction,
                    solimp,
                    geoms,
                    collision_pairid,
                    worldid,
                    contact_dist_out,
                    contact_pos_out,
                    contact_frame_out,
                    contact_includemargin_out,
                    contact_friction_out,
                    contact_solref_out,
                    contact_solreffriction_out,
                    contact_solimp_out,
                    contact_dim_out,
                    contact_geom_out,
                    contact_efc_address_out,
                    contact_worldid_out,
                    contact_type_out,
                    contact_geomcollisionid_out,
                    nacon_out,
                )

                dist_min1 = wp.cross(min_normal, min_pos - pos1)

                id2 = int(-1)
                dist_12 = float(-MJ_MAXVAL)
                for i in range(count):
                    if i == min_id or i == id1:
                        continue
                    hf_pos_i = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
                    dist = wp.abs(wp.dot(hf_pos_i - min_pos, dist_min1))
                    if dist > dist_12:
                        id2 = i
                        dist_12 = dist

                if id2 == -1 or (0.0 < dist_12 and dist_12 < MIN_DIST_TO_NEXT_CONTACT):
                    return

                pos2 = wp.vec3(hfield_contact_pos[id2, 0], hfield_contact_pos[id2, 1], hfield_contact_pos[id2, 2])
                normal2 = wp.vec3(hfield_contact_normal[id2, 0], hfield_contact_normal[id2, 1], hfield_contact_normal[id2, 2])

                write_contact(
                    naconmax_in,
                    2,
                    hfield_contact_dist[id2],
                    pos2,
                    make_frame(normal2),
                    margin,
                    gap,
                    condim,
                    friction,
                    solref,
                    solreffriction,
                    solimp,
                    geoms,
                    collision_pairid,
                    worldid,
                    contact_dist_out,
                    contact_pos_out,
                    contact_frame_out,
                    contact_includemargin_out,
                    contact_friction_out,
                    contact_solref_out,
                    contact_solreffriction_out,
                    contact_solimp_out,
                    contact_dim_out,
                    contact_geom_out,
                    contact_efc_address_out,
                    contact_worldid_out,
                    contact_type_out,
                    contact_geomcollisionid_out,
                    nacon_out,
                )

                vec_min2 = wp.cross(min_normal, min_pos - pos2)
                vec_12 = wp.cross(min_normal, pos1 - pos2)

                id3 = int(-1)
                dist3 = float(-MJ_MAXVAL)
                for i in range(count):
                    if i == min_id or i == id1 or i == id2:
                        continue
                    hf_pos_i = wp.vec3(hfield_contact_pos[i, 0], hfield_contact_pos[i, 1], hfield_contact_pos[i, 2])
                    dist = wp.abs(wp.dot(hf_pos_i - min_pos, vec_min2)) + wp.abs(wp.dot(pos1 - hf_pos_i, vec_12))
                    if dist > dist3:
                        id3 = i
                        dist3 = dist

                if id3 == -1 or (0.0 < dist3 and dist3 < MIN_DIST_TO_NEXT_CONTACT):
                    return

                pos3 = wp.vec3(hfield_contact_pos[id3, 0], hfield_contact_pos[id3, 1], hfield_contact_pos[id3, 2])
                normal3 = wp.vec3(hfield_contact_normal[id3, 0], hfield_contact_normal[id3, 1], hfield_contact_normal[id3, 2])

                write_contact(
                    naconmax_in,
                    3,
                    hfield_contact_dist[id3],
                    pos3,
                    make_frame(normal3),
                    margin,
                    gap,
                    condim,
                    friction,
                    solref,
                    solreffriction,
                    solimp,
                    geoms,
                    collision_pairid,
                    worldid,
                    contact_dist_out,
                    contact_pos_out,
                    contact_frame_out,
                    contact_includemargin_out,
                    contact_friction_out,
                    contact_solref_out,
                    contact_solreffriction_out,
                    contact_solimp_out,
                    contact_dim_out,
                    contact_geom_out,
                    contact_efc_address_out,
                    contact_worldid_out,
                    contact_type_out,
                    contact_geomcollisionid_out,
                    nacon_out,
                )

        return ccd_hfield_kernel

    # Monkey-patch the module so the call site in collision_convex.py picks up the new builder
    _cc.ccd_hfield_kernel_builder = ccd_hfield_kernel_builder_perenv
