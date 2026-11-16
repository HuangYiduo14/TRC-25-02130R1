"""Predeclared demand strata and exact SHA256 sampling without Python row loops."""
import hashlib
import numpy as np
import pandas as pd
from numba import njit

K = np.array([
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2],dtype=np.uint32)
INITIAL = np.array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19],dtype=np.uint32)


@njit(cache=True, inline='always')
def rotate(x, n):
    return np.uint32((x >> n) | (x << (32 - n)))


@njit(cache=True, nogil=True)
def od_sha256(origin, destination, uid_bytes, lengths, seed):
    """Exact SHA256(seed + origin_uid + '|' + destination_uid), one-block guard."""
    result = np.empty((len(origin), 8), np.uint32)
    for row in range(len(origin)):
        u, v = origin[row], destination[row]
        length = len(seed) + lengths[u] + 1 + lengths[v]
        if length >= 56:
            raise ValueError('Stable OD key exceeds validated single-block SHA256 domain')
        block = np.zeros(64, np.uint8)
        block[:len(seed)] = seed
        j = len(seed)
        for i in range(lengths[u]):
            block[j] = uid_bytes[u, i]; j += 1
        block[j] = 124; j += 1
        for i in range(lengths[v]):
            block[j] = uid_bytes[v, i]; j += 1
        block[j] = 128
        block[62] = (length * 8) >> 8
        block[63] = (length * 8) & 255
        w = np.empty(64, np.uint32)
        for i in range(16):
            w[i] = np.uint32(block[4*i]) << 24 | np.uint32(block[4*i+1]) << 16 | np.uint32(block[4*i+2]) << 8 | np.uint32(block[4*i+3])
        for i in range(16, 64):
            x, y = w[i-15], w[i-2]
            s0 = rotate(x,7) ^ rotate(x,18) ^ (x >> 3)
            s1 = rotate(y,17) ^ rotate(y,19) ^ (y >> 10)
            w[i] = np.uint32(w[i-16] + s0 + w[i-7] + s1)
        a,b,c,d,e,f,g,h = INITIAL
        for i in range(64):
            s1 = rotate(e,6) ^ rotate(e,11) ^ rotate(e,25)
            ch = (e & f) ^ ((~e) & g)
            t1 = np.uint32(h + s1 + ch + K[i] + w[i])
            s0 = rotate(a,2) ^ rotate(a,13) ^ rotate(a,22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = np.uint32(s0 + maj)
            h,g,f,e,d,c,b,a = g,f,e,np.uint32(d+t1),c,b,a,np.uint32(t1+t2)
        final = (a,b,c,d,e,f,g,h)
        for i in range(8):
            result[row,i] = np.uint32(INITIAL[i] + final[i])
    return result


def encode_uids(uids):
    text = np.asarray(uids, str)
    if any('|' in s for s in text):
        raise ValueError('Ambiguous OD key delimiter in accepted UID')
    encoded = np.char.encode(text, 'utf-8')
    return encoded.view(np.uint8).reshape(len(text), -1), np.char.str_len(encoded).astype(np.int64)


def freeze_sample(b, old, supply, seed, mark=lambda x: None):
    od, g = b.od, b.graph
    origin = np.repeat(od.origin_group_node_idx, np.diff(od.origin_offsets))
    dest = od.destination_node_idx
    if supply.cross_scc_reachable_row_count != 0:
        raise ValueError('Accepted cache contains cross-SCC reachable ODs; exact mask required before sampling')
    reach = supply.scc_labels[origin] == supply.scc_labels[dest]
    if int((~reach).sum()) != supply.unreachable_row_count:
        raise ValueError('Accepted SCC row coverage differs')
    keys = origin.astype(np.uint64) * np.uint64(len(g.node_idx_to_uid)) + dest.astype(np.uint64)
    oldkeys = old.origin_idx.to_numpy(np.uint64) * np.uint64(len(g.node_idx_to_uid)) + old.destination_idx.to_numpy(np.uint64)
    positions = np.argsort(keys, kind='stable')
    oldrow = positions[np.searchsorted(keys[positions], oldkeys)]
    if not np.array_equal(keys[oldrow], oldkeys) or not np.all(reach[oldrow]) or len(np.unique(oldrow)) != 200:
        raise ValueError('Old 200 identities fail accepted OD lookup')
    eligible = np.flatnonzero(reach & (od.demand > 0))
    eligible = eligible[~np.isin(eligible, oldrow)]
    uid_order = np.argsort(g.node_idx_to_uid.astype(str), kind='stable')
    rank = np.empty(len(uid_order), np.int64); rank[uid_order] = np.arange(len(rank))
    order = np.lexsort((rank[dest[eligible]], rank[origin[eligible]], od.demand[eligible]))
    strata = np.array_split(eligible[order], 8)
    data, lengths = encode_uids(g.node_idx_to_uid)
    seedbytes = np.frombuffer(seed.encode(), np.uint8)
    chosen = []
    rows = []
    for h, indexes in enumerate(strata):
        mark(f'F1 sampling stratum {h+1}/8: {len(indexes)} reachable positive-demand rows')
        digests = od_sha256(origin[indexes], dest[indexes], data, lengths, seedbytes)
        # Structured digest ordering compares all eight big-endian logical words.
        sort = np.lexsort(tuple(digests[:, j] for j in range(7, -1, -1)))
        take = min(32, len(indexes))
        for j in range(take):
            r = int(indexes[sort[j]])
            digest = ''.join(f'{int(v):08x}' for v in digests[sort[j]])
            key = str(g.node_idx_to_uid[origin[r]]) + '|' + str(g.node_idx_to_uid[dest[r]])
            if digest != hashlib.sha256((seed + key).encode()).hexdigest():
                raise ValueError('Compiled SHA256 differs from independent hashlib')
            chosen.append(r)
            role = 'NEW_SELECT128' if j < min(16, take) else 'NEW_AUDIT128'
            nh = min(16, take) if role == 'NEW_SELECT128' else max(0, take-16)
            rows.append(dict(accepted_od_row=r, stable_od_key=key, source_set=role, stratum=h,
                N_h=len(indexes), n_h=nh, inclusion_fraction=nh/len(indexes), selection_sha256=digest,
                source_sample_od_id=''))
    if len(chosen) != 256:
        raise ValueError('Small-stratum redistribution required; do not silently reduce sample')
    for r, sid in zip(oldrow, old.sample_od_id):
        key = str(g.node_idx_to_uid[origin[r]]) + '|' + str(g.node_idx_to_uid[dest[r]])
        rows.append(dict(accepted_od_row=int(r), stable_od_key=key, source_set='DISCOVERY_OLD200',
            stratum=-1, N_h=200, n_h=200, inclusion_fraction=1., selection_sha256='', source_sample_od_id=sid))
    sample = pd.DataFrame(rows)
    rr = sample.accepted_od_row.to_numpy()
    sample['origin_idx'] = origin[rr]; sample['destination_idx'] = dest[rr]
    sample['origin_uid'] = g.node_idx_to_uid[origin[rr]].astype(str)
    sample['destination_uid'] = g.node_idx_to_uid[dest[rr]].astype(str)
    sample['demand'] = od.demand[rr]
    sample['ratio_weight'] = sample.demand / sample.inclusion_fraction
    sample = sample.sort_values(['source_set','stratum','stable_od_key'], kind='stable').reset_index(drop=True)
    if sample.stable_od_key.duplicated().any() or len(sample) != 456:
        raise ValueError('Frozen sample coverage/independence failed')
    return sample
