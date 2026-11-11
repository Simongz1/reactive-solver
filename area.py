#!/usr/bin/env python3
"""Area of regions where a field exceeds a threshold in a 2D Exodus II file.

Nodal ("point-wise") variables: every element is split into linear triangles
(quads along the 0-2 diagonal) and each triangle is clipped exactly against
the isoline f = threshold, so partially covered elements contribute only
their covered fraction. Element variables: an element counts in full when
its value passes the threshold.

Connected regions: elements are joined when they share a node that lies in
the region (above threshold for nodal variables; any node of two selected
elements for element variables). Each region reports area and centroid.

Usage examples:
  python exodus_threshold_area.py out.e --var pressure --threshold 1e9
  python exodus_threshold_area.py out.e --var rho --threshold 500 --below
  python exodus_threshold_area.py out.e --var T --threshold 800 --all-steps --csv areas.csv
  python exodus_threshold_area.py out.e --list

Requires numpy and netCDF4 (pip install numpy netCDF4). scipy is optional
(used for faster region labelling on large meshes).
"""
import argparse
import csv
import sys

import numpy as np

try:
    from netCDF4 import Dataset, chartostring
except ImportError:
    sys.exit('This script needs netCDF4: pip install netCDF4 numpy')


# ----------------------------------------------------------------- reading --

def _names(ds, var):
    if var not in ds.variables:
        return []
    return [str(s).strip() for s in chartostring(ds.variables[var][:])]


class ExodusMesh:
    """Minimal read-only view of a 2D Exodus II file."""

    def __init__(self, path):
        self.ds = Dataset(path, 'r')
        ds = self.ds
        self.ndim = len(ds.dimensions['num_dim'])
        if 'coordx' in ds.variables:
            x = ds.variables['coordx'][:]
            y = ds.variables['coordy'][:]
        else:  # old single-array layout
            c = ds.variables['coord'][:]
            x, y = c[0], c[1]
        self.xy = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        if self.ndim == 3 and 'coordz' in ds.variables:
            z = np.asarray(ds.variables['coordz'][:], float)
            if np.ptp(z) > 1e-12 * max(np.ptp(x), np.ptp(y), 1.0):
                print('warning: mesh has non-zero z extent; areas are computed '
                      'in the x-y projection', file=sys.stderr)

        nblk = len(ds.dimensions['num_el_blk']) if 'num_el_blk' in ds.dimensions else 0
        ids = ds.variables['eb_prop1'][:] if 'eb_prop1' in ds.variables else range(1, nblk + 1)
        names = _names(ds, 'eb_names') or [''] * nblk
        self.blocks = []
        for j in range(nblk):
            key = f'connect{j + 1}'
            if key not in ds.variables:
                continue  # empty block
            conn_var = ds.variables[key]
            etype = getattr(conn_var, 'elem_type', 'UNKNOWN').upper()
            conn = np.asarray(conn_var[:], dtype=np.int64) - 1  # 0-based
            self.blocks.append(dict(index=j + 1, id=int(ids[j]),
                                    name=names[j] or f'block_{int(ids[j])}',
                                    type=etype, conn=conn))

        self.times = (np.asarray(ds.variables['time_whole'][:], float)
                      if 'time_whole' in ds.variables else np.array([0.0]))
        self.nodal_names = _names(ds, 'name_nod_var')
        self.elem_names = _names(ds, 'name_elem_var')

    def nodal(self, name, step):
        i = self.nodal_names.index(name)
        if f'vals_nod_var{i + 1}' in self.ds.variables:
            v = self.ds.variables[f'vals_nod_var{i + 1}'][step]
        else:  # old layout: (time, var, node)
            v = self.ds.variables['vals_nod_var'][step, i]
        return np.ma.filled(np.asarray(v, float), np.nan)

    def element(self, name, step, block):
        i = self.elem_names.index(name)
        key = f'vals_elem_var{i + 1}eb{block["index"]}'
        if key not in self.ds.variables:  # variable not defined on block
            return None
        return np.ma.filled(np.asarray(self.ds.variables[key][step], float), np.nan)


# -------------------------------------------------------------- geometry --

def corner_triangles(block):
    """Split a block's elements into linear triangles of corner nodes.

    Returns (tri_nodes[m, 3], tri_elem[m]) with tri_elem the local element
    index each triangle came from.
    """
    conn = block['conn']
    n = len(conn)
    if n_corners(block) == 3:
        return conn[:, :3], np.arange(n)
    c = conn[:, :4]
    tris = np.concatenate([c[:, [0, 1, 2]], c[:, [0, 2, 3]]])
    return tris, np.concatenate([np.arange(n), np.arange(n)])


def n_corners(block):
    """3 for triangles, 4 for quads (higher-order mid-side nodes are ignored)."""
    etype, npe = block['type'], block['conn'].shape[1]
    if etype.startswith('TRI') or (npe in (3, 6, 7) and not etype.startswith(('QUAD', 'SHELL'))):
        return 3
    if etype.startswith(('QUAD', 'SHELL')) or npe in (4, 8, 9):
        return 4
    raise ValueError(f'block {block["name"]}: element type {etype} is not a 2D '
                     'triangle/quad; only 2D meshes are supported')


def tri_area_centroid(p0, p1, p2):
    a = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                     - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return a, (p0 + p1 + p2) / 3.0


def clipped_tri(p, s):
    """Area and centroid of {s > 0} for linear s on triangles.

    p: (m, 3, 2) vertex coordinates; s: (m, 3) values minus threshold.
    """
    m = len(s)
    area = np.zeros(m)
    cen = np.zeros((m, 2))
    pos = s > 0
    npos = pos.sum(1)

    full_a, full_c = tri_area_centroid(p[:, 0], p[:, 1], p[:, 2])
    k = npos == 3
    area[k], cen[k] = full_a[k], full_c[k]

    # One vertex on the odd side: the positive part (npos == 1) or the
    # negative corner to remove (npos == 2) is a small triangle at that vertex.
    for want, sign in ((1, 1.0), (2, -1.0)):
        k = np.where(npos == want)[0]
        if not len(k):
            continue
        odd_is_pos = pos[k] if want == 1 else ~pos[k]
        a = np.argmax(odd_is_pos, axis=1)
        b, c = (a + 1) % 3, (a + 2) % 3
        pa, pb, pc = p[k, a], p[k, b], p[k, c]
        sa, sb, sc = s[k, a], s[k, b], s[k, c]
        tb = sa / (sa - sb)
        tc = sa / (sa - sc)
        qb = pa + tb[:, None] * (pb - pa)
        qc = pa + tc[:, None] * (pc - pa)
        corner_a, corner_c = tri_area_centroid(pa, qb, qc)
        if sign > 0:
            area[k], cen[k] = corner_a, corner_c
        else:
            rem = full_a[k] - corner_a
            safe = np.where(rem > 0, rem, 1.0)
            area[k] = rem
            cen[k] = (full_a[k, None] * full_c[k] - corner_a[:, None] * corner_c) / safe[:, None]
    return area, cen


# --------------------------------------------------------------- labelling --

def connected_labels(n_vertices, edges_a, edges_b):
    """Connected-component label for each vertex of an undirected graph."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(edges_a)), (edges_a, edges_b)),
                       shape=(n_vertices, n_vertices))
        return connected_components(g, directed=False)[1]
    except ImportError:
        parent = np.arange(n_vertices)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a, b in zip(edges_a.tolist(), edges_b.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        return np.array([find(i) for i in range(n_vertices)])


# ---------------------------------------------------------------- analysis --

def analyse_step(mesh, var, kind, threshold, step, below, block_filter):
    """Return list of regions [{area, cx, cy, n_elem}] sorted by area."""
    xy = mesh.xy
    n_nodes = len(xy)
    nodal = mesh.nodal(var, step) if kind == 'nodal' else None

    elem_area, elem_cen, elem_conn, elem_active_nodes = [], [], [], []
    for blk in mesh.blocks:
        if block_filter and blk['name'] not in block_filter and str(blk['id']) not in block_filter:
            continue
        tris, tri_elem = corner_triangles(blk)
        n_el = len(blk['conn'])
        if kind == 'nodal':
            s = nodal[tris] - threshold
            if below:
                s = -s
            s = np.where(np.isnan(s), -1.0, s)
            # Exactly-at-threshold vertices are treated as outside; nudge to
            # avoid 0/0 when an edge has both ends on the isoline.
            s = np.where(s == 0, -np.finfo(float).tiny, s)
            ta, tc = clipped_tri(xy[tris], s)
            node_in = (nodal < threshold) if below else (nodal > threshold)
            act = node_in[blk['conn'][:, :n_corners(blk)]]
        else:
            ev = mesh.element(var, step, blk)
            if ev is None:
                continue
            sel = (ev < threshold) if below else (ev > threshold)
            full_a, full_c = tri_area_centroid(*(xy[tris[:, i]] for i in range(3)))
            ta = np.where(sel[tri_elem], full_a, 0.0)
            tc = full_c
            act = np.repeat(sel[:, None], n_corners(blk), axis=1)

        a = np.bincount(tri_elem, weights=ta, minlength=n_el)
        cx = np.bincount(tri_elem, weights=ta * tc[:, 0], minlength=n_el)
        cy = np.bincount(tri_elem, weights=ta * tc[:, 1], minlength=n_el)
        elem_area.append(a)
        elem_cen.append(np.column_stack([cx, cy]))  # area-weighted sums
        elem_conn.append(blk['conn'][:, :act.shape[1]])
        elem_active_nodes.append(act)

    if not elem_area:
        return []
    area = np.concatenate(elem_area)
    wsum = np.concatenate(elem_cen)
    keep = area > 0
    if not keep.any():
        return []

    # Graph on [elements | nodes]: link each kept element to its in-region
    # nodes, so elements sharing such a node fall in one component.
    n_el = len(area)
    ea, eb = [], []
    offset = 0
    for conn, act in zip(elem_conn, elem_active_nodes):
        idx = np.arange(len(conn)) + offset
        ok = act & keep[idx][:, None]
        rows, cols = np.nonzero(ok)
        ea.append(idx[rows])
        eb.append(n_el + conn[rows, cols])
        offset += len(conn)
    lab = connected_labels(n_el + n_nodes, np.concatenate(ea), np.concatenate(eb))[:n_el]

    lab_k = lab[keep]
    uniq, inv = np.unique(lab_k, return_inverse=True)
    r_area = np.bincount(inv, weights=area[keep])
    r_cx = np.bincount(inv, weights=wsum[keep, 0]) / r_area
    r_cy = np.bincount(inv, weights=wsum[keep, 1]) / r_area
    r_n = np.bincount(inv)
    order = np.argsort(-r_area)
    return [dict(area=r_area[i], cx=r_cx[i], cy=r_cy[i], n_elem=int(r_n[i])) for i in order]


# --------------------------------------------------------------------- CLI --

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('exodus', help='Exodus II file (.e, .exo, .ex2)')
    ap.add_argument('--var', help='variable name (nodal or element)')
    ap.add_argument('--threshold', type=float, help='threshold value')
    ap.add_argument('--below', action='store_true',
                    help='measure regions where var < threshold instead of >')
    ap.add_argument('--element-var', action='store_true',
                    help='force treating --var as an element variable')
    ap.add_argument('--step', type=int, default=-1,
                    help='time-step index (0-based, negative from end; default last)')
    ap.add_argument('--all-steps', action='store_true', help='process every time step')
    ap.add_argument('--blocks', nargs='+', help='restrict to these block names or ids')
    ap.add_argument('--min-area', type=float, default=0.0,
                    help='ignore regions smaller than this area')
    ap.add_argument('--csv', help='write per-region results to this CSV file')
    ap.add_argument('--list', action='store_true', help='list variables, blocks and times, then exit')
    args = ap.parse_args()

    mesh = ExodusMesh(args.exodus)
    if args.list:
        print(f'dimension: {mesh.ndim}, nodes: {len(mesh.xy)}, time steps: {len(mesh.times)}')
        for b in mesh.blocks:
            print(f'  block {b["id"]} "{b["name"]}": {len(b["conn"])} x {b["type"]}')
        print('nodal variables:  ', ', '.join(mesh.nodal_names) or '(none)')
        print('element variables:', ', '.join(mesh.elem_names) or '(none)')
        if len(mesh.times):
            print(f'times: {mesh.times[0]:g} .. {mesh.times[-1]:g}')
        return
    if args.var is None or args.threshold is None:
        ap.error('--var and --threshold are required (or use --list)')

    if args.var in mesh.nodal_names and not args.element_var:
        kind = 'nodal'
    elif args.var in mesh.elem_names:
        kind = 'element'
    else:
        ap.error(f'variable "{args.var}" not found. Nodal: {mesh.nodal_names}; '
                 f'element: {mesh.elem_names}')

    nt = len(mesh.times)
    steps = range(nt) if args.all_steps else [args.step % nt]
    cmp = '<' if args.below else '>'
    rows = []
    for st in steps:
        regions = [r for r in analyse_step(mesh, args.var, kind, args.threshold, st,
                                           args.below, args.blocks)
                   if r['area'] >= args.min_area]
        total = sum(r['area'] for r in regions)
        print(f'step {st}  t = {mesh.times[st]:.6g}  {args.var} {cmp} {args.threshold:g} '
              f'({kind}): total area = {total:.6g} in {len(regions)} region(s)')
        for i, r in enumerate(regions):
            if not args.all_steps or i < 5:
                print(f'    region {i}: area = {r["area"]:.6g}  centroid = '
                      f'({r["cx"]:.6g}, {r["cy"]:.6g})  elements = {r["n_elem"]}')
            rows.append([st, mesh.times[st], i, r['area'], r['cx'], r['cy'], r['n_elem'], total])
        if args.all_steps and len(regions) > 5:
            print(f'    ... {len(regions) - 5} more')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'time', 'region', 'area', 'centroid_x', 'centroid_y',
                        'n_elements', 'total_area'])
            w.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
